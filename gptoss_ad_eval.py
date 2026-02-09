import pandas as pd
import multiprocessing as mp
import numpy as np
import ollama
import tiktoken
import os
import json
import logging
import time
import re
from datetime import datetime
from tqdm import tqdm
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')


# Set up logging
def setup_logging():
    log_filename = f"ad_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# Set the environment variables
os.environ["OLLAMA_NUM_PARALLEL"] = "2"
os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
os.environ["OLLAMA_SCHED_SPREAD"] = "1"

# Create output directory for temporary files
output_dir = "ad_evaluation_results"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"Created output directory: {output_dir}")


def create_evaluation_prompt(headline, description, context=None):
    """
    Creates the evaluation prompt for LLM-based ad quality assessment
    """
    context_str = context if context else "Study Economics"

    prompt = f"""# ROLE DEFINITION
You are Senior Digital Marketing Director with 15+ years of experience evaluating sponsored search ads (Google Ads, Bing Ads). You specialize in judging ad quality for search engine performance.

Your expertise includes:
- Understanding user search intent and query relevance
- Evaluating persuasive copywriting and value proposition clarity
- Assessing call-to-action effectiveness
- Recognizing brand voice consistency and trust signals
- Identifying spammy or low-quality content

# TASK INSTRUCTIONS
You will evaluate a sponsored search advertisement based on the following **five criteria**:

1. **SEARCH INTENT ALIGNMENT** (0-20 points)
   - How well does the ad match what a user is searching for?
   - Does it directly address the user's likely needs/goals?
   - Is it relevant to the search query context?

2. **PERSUASIVE COPY QUALITY** (0-20 points)
   - How compelling and engaging is the language?
   - Does it highlight clear benefits or unique selling points?
   - Is the messaging concise yet informative?
   - Does it create urgency or interest?

3. **CALL-TO-ACTION EFFECTIVENESS** (0-20 points)
   - Is there a clear, compelling next step for the user?
   - Does the CTA feel natural and contextually appropriate?
   - Would it motivate a searcher to click?

4. **PROFESSIONALISM & TRUST** (0-20 points)
   - Does the ad sound professional and credible?
   - Are there any spammy, exaggerated, or misleading claims?
   - Would this ad build or erode trust in the brand?

5. **OVERALL PERFORMANCE POTENTIAL** (0-20 points)
   - Based on your expert judgment, how likely is this ad to achieve:
     * High click-through rate (CTR)?
     * Good conversion potential?
     * Positive user experience?

**SCORING SYSTEM:**
- Each criterion: 0-20 points (0=poor, 10=average, 20=excellent)
- Total score: 0-100 points
- **Thresholds:** <40=Poor, 40-59=Average, 60-79=Good, 80-100=Excellent

# INPUT FORMAT
The ad will be provided in this format:
**Headline:** [Advertisement headline text]
**Description:** [Advertisement description text]
**Context:** [Optional: Search query/keyword context if available]

# OUTPUT FORMAT REQUIREMENTS
**CRITICAL:** You must output ONLY a JSON object with exactly this structure:
{{
  "scores": {{
    "search_intent_alignment": [0-20],
    "persuasive_copy_quality": [0-20],
    "cta_effectiveness": [0-20],
    "professionalism_trust": [0-20],
    "overall_potential": [0-20],
    "total_score": [0-100]
  }},
  "grade": ["Poor"|"Average"|"Good"|"Excellent"],
  "justification": {{
    "search_intent": "[1-2 sentence explanation of score]",
    "copy_quality": "[1-2 sentence explanation of score]",
    "cta": "[1-2 sentence explanation of score]",
    "trust": "[1-2 sentence explanation of score]",
    "overall": "[1-2 sentence summary of why this ad would/wouldn't perform well]"
  }}
}}

# EVALUATION EXAMPLES

**Example 1 (High Quality):**
Input:
**Headline:** Harvard Business School - MBA Program
**Description:** Transform your career with our world-renowned MBA. Join 20,000+ alumni network. Applications open now.
**Context:** "MBA programs"

Output:
{{
  "scores": {{
    "search_intent_alignment": 18,
    "persuasive_copy_quality": 17,
    "cta_effectiveness": 16,
    "professionalism_trust": 19,
    "overall_potential": 18,
    "total_score": 88
  }},
  "grade": "Excellent",
  "justification": {{
    "search_intent": "Directly addresses MBA seekers with clear relevance to the query.",
    "copy_quality": "Strong benefit-focused language ('transform your career', 'world-renowned') with social proof.",
    "cta": "Clear action direction with urgency ('applications open now').",
    "trust": "Established brand name with alumni network evidence builds credibility.",
    "overall": "This ad would likely achieve high CTR and attract qualified applicants due to clear value proposition and strong brand appeal."
  }}
}}

**Example 2 (Low Quality):**
Input:
**Headline:** BEST MBA CHEAP ONLINE FAST
**Description:** Get MBA degree quick easy. No tests. Accredited. Click here!!!
**Context:** "MBA programs"

Output:
{{
  "scores": {{
    "search_intent_alignment": 12,
    "persuasive_copy_quality": 6,
    "cta_effectiveness": 8,
    "professionalism_trust": 4,
    "overall_potential": 7,
    "total_score": 37
  }},
  "grade": "Poor",
  "justification": {{
    "search_intent": "Matches keyword but emphasizes wrong benefits ('cheap', 'fast') over quality.",
    "copy_quality": "Spammy capitalization, vague claims, lacks substantive benefits.",
    "cta": "Generic 'click here' with excessive punctuation feels low-quality.",
    "trust": "'No tests' claim undermines credibility; feels like a diploma mill.",
    "overall": "Despite keyword matching, this ad would likely have low CTR from serious candidates and high bounce rates due to trust issues."
  }}
}}

# IMPORTANT INSTRUCTIONS
* **NEVER** ask follow-up questions - evaluate based only on the provided ad
* **ALWAYS** output valid JSON exactly as specified above
* **DO NOT** include any explanatory text outside the JSON
* **Base scores on marketing effectiveness**, not personal preferences
* If no context is provided, assume a generic search intent
* **Score consistently** across similar quality ads
* **Penalize** keyword stuffing, exaggeration, and unclear messaging
* **Reward** clear benefits, specificity, and user-focused language

# AD TO EVALUATE
**Headline:** {headline}
**Description:** {description}
**Context:** {context_str}"""

    return prompt


def clean_text(text, max_tokens=2000):
    """
    Clean ad text for LLM processing
    """
    if not text or pd.isna(text):
        return ""

    text_str = str(text)

    # Remove HTML tags if any
    text_str = re.sub(r'<[^>]*>', '', text_str)

    # Remove excessive whitespace
    text_str = re.sub(r'\s+', ' ', text_str).strip()

    # Truncate if too long
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text_str)
        if len(tokens) > max_tokens:
            text_str = encoding.decode(tokens[:max_tokens])
    except Exception as e:
        logger.warning(f"Tokenization error, using text as is: {e}")

    return text_str


def evaluate_ad(headline, description, context=None):
    """
    Use LLM to evaluate ad content quality
    """
    try:
        prompt = create_evaluation_prompt(headline, description, context)

        response = ollama.chat(
            model='gpt-oss:20b',
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'num_gpu': 25,
                "temperature": 0,
                "seed": 1,
                "repeat_last_n": 64,
                "repeat_penalty": 1.2,
                "num_ctx": 8192
            }
        )

        full_response = response["message"]["content"].strip()

        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                # Try to fix common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                try:
                    result = json.loads(json_str)
                    return result
                except:
                    logger.error(f"Failed to parse JSON after cleanup")
                    return {"error": "Failed to parse JSON response"}
        else:
            logger.error(f"No JSON found in response: {full_response[:200]}...")
            return {"error": "No JSON in response"}

    except Exception as e:
        logger.error(f"Error in LLM call: {e}")
        return {"error": str(e)}


def save_temp_result(chunk_id, result):
    """Save a single result to temporary JSON file"""
    try:
        chunk_filename = f"{output_dir}/ad_chunk_{chunk_id}.json"

        if os.path.exists(chunk_filename):
            try:
                with open(chunk_filename, 'r') as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_results = []
        else:
            existing_results = []

        existing_results.append(result)

        with open(chunk_filename, 'w') as f:
            json.dump(existing_results, f, indent=2)

        logger.debug(f"Chunk {chunk_id}: Appended result to {chunk_filename}")

    except Exception as e:
        logger.error(f"Error appending to chunk {chunk_id} file: {e}")


def get_processed_indices(chunk_id):
    """Get all indices that have already been processed for a chunk"""
    chunk_filename = f"{output_dir}/ad_chunk_{chunk_id}.json"
    processed_indices = set()

    if os.path.exists(chunk_filename):
        try:
            with open(chunk_filename, 'r') as f:
                chunk_data = json.load(f)

                for item in chunk_data:
                    if 'index' in item:
                        processed_indices.add(item['index'])

            logger.info(f"Chunk {chunk_id}: Found {len(processed_indices)} already processed rows")

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Chunk {chunk_id}: Error reading file - {e}")
            with open(chunk_filename, 'w') as f:
                json.dump([], f)

    return processed_indices


def find_column_indices(df):
    """
    Find the correct column indices for H (headline), D (description), and Group
    Based on the Excel file structure
    """
    headline_idx = None
    description_idx = None
    group_idx = None

    # Look for columns that might contain the data
    for i, col in enumerate(df.columns):
        col_str = str(col)
        # Check if this column contains headline data (look for ad text patterns)
        if 'MBA' in col_str or 'business school' in col_str.lower() or 'degree' in col_str.lower():
            if headline_idx is None:
                headline_idx = i
            continue

        # Check for description patterns
        if 'Choose among' in col_str or 'Study a' in col_str or 'programs' in col_str.lower():
            if description_idx is None and i != headline_idx:
                description_idx = i
            continue

        # Check for group column
        if 'PPLM' in col_str or 'Human' in col_str:
            group_idx = i

    # If we couldn't find by pattern, use the last 3 columns as fallback
    if headline_idx is None or description_idx is None or group_idx is None:
        logger.warning("Could not find columns by pattern, using last 3 columns as fallback")
        headline_idx = len(df.columns) - 3
        description_idx = len(df.columns) - 2
        group_idx = len(df.columns) - 1

    logger.info(f"Column indices found: Headline={headline_idx}, Description={description_idx}, Group={group_idx}")
    return headline_idx, description_idx, group_idx


def label_chunk(df_chunk, process_id, processed_indices, column_indices):
    """
    Process a chunk of ads for evaluation
    """
    results = []
    row_count = 0
    start_time = time.time()

    headline_idx, description_idx, group_idx = column_indices

    # Get unprocessed rows
    unprocessed_rows = df_chunk[~df_chunk.index.isin(processed_indices)]
    total_rows = len(unprocessed_rows)

    if total_rows == 0:
        logger.info(f"Process {process_id}: All rows already processed, skipping")
        print(f"[Chunk {process_id}] All rows already processed, skipping")
        return results

    logger.info(f"Chunk {process_id}: Processing {total_rows} unprocessed rows")
    print(f"[Chunk {process_id}] Starting processing of {total_rows} unprocessed rows...")

    for index, row in unprocessed_rows.iterrows():
        try:
            # Extract data using the found column indices
            headline = clean_text(row.iloc[headline_idx]) if len(row) > headline_idx else ""
            description = clean_text(row.iloc[description_idx]) if len(row) > description_idx else ""
            group = str(row.iloc[group_idx]) if len(row) > group_idx else "Unknown"

            # Debug output for first few rows
            if row_count < 3:
                logger.info(
                    f"Row {index}: Headline='{headline[:50]}...', Description='{description[:50]}...', Group='{group}'")

            # Evaluate ad
            evaluation = evaluate_ad(headline, description)

            # Prepare result
            result_data = {
                'index': index,
                'headline': headline,
                'description': description,
                'group': group,
                'evaluation': evaluation,
                'timestamp': datetime.now().isoformat(),
                'chunk_id': process_id,
                'result_number': row_count + 1
            }

            # Extract score if available
            if isinstance(evaluation, dict) and 'scores' in evaluation:
                result_data['total_score'] = evaluation['scores'].get('total_score', 0)
                result_data['grade'] = evaluation.get('grade', 'Unknown')

            results.append(result_data)
            save_temp_result(process_id, result_data)

            # Progress update
            row_count += 1
            if row_count % 2 == 0 or row_count == total_rows:
                elapsed = time.time() - start_time
                rate = row_count / elapsed if elapsed > 0.1 else 0
                print(f"[Chunk {process_id}] Processed {row_count}/{total_rows} rows "
                      f"({row_count / total_rows * 100:.1f}%) - Rate: {rate:.2f} rows/sec")

        except Exception as e:
            logger.error(f"Error processing row {index} in process {process_id}: {e}")
            result_data = {
                'index': index,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'chunk_id': process_id,
                'result_number': row_count + 1
            }
            results.append(result_data)
            save_temp_result(process_id, result_data)
            row_count += 1

    elapsed_time = time.time() - start_time
    processing_rate = row_count / elapsed_time if elapsed_time > 0.1 else row_count

    print(f"[Chunk {process_id}] COMPLETED: {row_count} rows in {elapsed_time:.2f}s "
          f"({processing_rate:.2f} rows/sec)")

    return results


def run_multiprocessing(df, num_processes=None):
    """
    Run ad evaluation in parallel
    """
    if num_processes is None:
        num_processes = 3

    logger.info(f"Using {num_processes} processes for parallel processing")

    # Find column indices first
    column_indices = find_column_indices(df)

    # Split the DataFrame into chunks
    df_split = np.array_split(df, num_processes)
    logger.info(f"Data split into {len(df_split)} chunks")

    # Get processed indices for each chunk
    processed_indices_by_chunk = {}
    for chunk_id in range(len(df_split)):
        processed_indices_by_chunk[chunk_id] = get_processed_indices(chunk_id)

    # Prepare arguments for multiprocessing
    args_list = []
    for chunk_id, chunk_data in enumerate(df_split):
        processed_indices = processed_indices_by_chunk[chunk_id]
        args_list.append((chunk_data, chunk_id, processed_indices, column_indices))

    try:
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(label_chunk, args_list)

        # Flatten results
        final_results = [item for sublist in results for item in sublist]

        print(f"\n{'=' * 60}")
        print(f"All chunks completed! Total results: {len(final_results)}")
        logger.info(f"All processes completed. Total results: {len(final_results)}")

        return final_results

    except Exception as e:
        logger.error(f"Error in multiprocessing: {e}")
        print(f"Multiprocessing failed: {e}")
        return []


def perform_t_test(results_df):
    """
    Perform t-test between PPLM and Human groups
    """
    print("\n" + "=" * 60)
    print("PERFORMING T-TEST ANALYSIS")
    print("=" * 60)

    # Extract scores by group
    pplm_scores = []
    human_scores = []

    for idx, row in results_df.iterrows():
        if 'evaluation' in row and isinstance(row['evaluation'], dict):
            score = row['evaluation'].get('scores', {}).get('total_score')
            group = row['group']

            if score is not None and isinstance(score, (int, float)):
                if 'PPLM' in str(group):
                    pplm_scores.append(score)
                elif 'Human' in str(group):
                    human_scores.append(score)

    print(f"\nGroup Statistics:")
    print(f"PPLM Group: n={len(pplm_scores)}, Mean={np.mean(pplm_scores):.2f}, SD={np.std(pplm_scores):.2f}")
    print(f"Human Group: n={len(human_scores)}, Mean={np.mean(human_scores):.2f}, SD={np.std(human_scores):.2f}")

    if len(pplm_scores) > 1 and len(human_scores) > 1:
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(pplm_scores, human_scores, equal_var=False)

        print(f"\nT-test Results:")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(pplm_scores) + np.var(human_scores)) / 2)
        cohens_d = (np.mean(pplm_scores) - np.mean(human_scores)) / pooled_std

        print(f"Cohen's d (effect size): {cohens_d:.4f}")

        # Interpretation
        print(f"\nInterpretation:")
        if p_value < 0.05:
            if np.mean(pplm_scores) > np.mean(human_scores):
                print("✓ STATISTICALLY SIGNIFICANT: PPLM ads scored significantly higher than Human ads")
            else:
                print("✓ STATISTICALLY SIGNIFICANT: Human ads scored significantly higher than PPLM ads")
        else:
            print("✗ NOT STATISTICALLY SIGNIFICANT: No significant difference between groups")

        # Calculate confidence intervals
        ci_pplm = stats.t.interval(0.95, len(pplm_scores) - 1,
                                   loc=np.mean(pplm_scores),
                                   scale=stats.sem(pplm_scores))
        ci_human = stats.t.interval(0.95, len(human_scores) - 1,
                                    loc=np.mean(human_scores),
                                    scale=stats.sem(human_scores))

        print(f"\n95% Confidence Intervals:")
        print(f"PPLM: ({ci_pplm[0]:.2f}, {ci_pplm[1]:.2f})")
        print(f"Human: ({ci_human[0]:.2f}, {ci_human[1]:.2f})")

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'pplm_mean': np.mean(pplm_scores),
            'human_mean': np.mean(human_scores),
            'pplm_n': len(pplm_scores),
            'human_n': len(human_scores)
        }
    else:
        print("Insufficient data for t-test")
        return None


def generate_summary_report(results_df, t_test_results=None):
    """
    Generate a comprehensive summary report
    """
    report = []
    report.append("=" * 80)
    report.append("AD EVALUATION SUMMARY REPORT")
    report.append("=" * 80)

    # Basic statistics
    total_ads = len(results_df)
    pplm_ads = len(results_df[results_df['group'].str.contains('PPLM', na=False)])
    human_ads = len(results_df[results_df['group'].str.contains('Human', na=False)])

    report.append(f"\nTotal Ads Evaluated: {total_ads}")
    report.append(f"PPLM Ads: {pplm_ads}")
    report.append(f"Human Ads: {human_ads}")

    # Score distribution by grade
    if 'grade' in results_df.columns:
        grade_counts = results_df['grade'].value_counts()
        report.append(f"\nGrade Distribution:")
        for grade, count in grade_counts.items():
            report.append(f"  {grade}: {count} ads ({count / total_ads * 100:.1f}%)")

    # Average scores by group
    if 'total_score' in results_df.columns:
        pplm_avg = results_df[results_df['group'].str.contains('PPLM', na=False)]['total_score'].mean()
        human_avg = results_df[results_df['group'].str.contains('Human', na=False)]['total_score'].mean()

        report.append(f"\nAverage Scores:")
        report.append(f"  PPLM: {pplm_avg:.2f}")
        report.append(f"  Human: {human_avg:.2f}")

    # T-test results
    if t_test_results:
        report.append(f"\nStatistical Analysis Results:")
        report.append(f"  t-statistic: {t_test_results['t_statistic']:.4f}")
        report.append(f"  p-value: {t_test_results['p_value']:.6f}")
        report.append(f"  Cohen's d: {t_test_results['cohens_d']:.4f}")
        report.append(f"  Sample sizes - PPLM: {t_test_results['pplm_n']}, Human: {t_test_results['human_n']}")

    report.append("\n" + "=" * 80)

    # Save report to file
    report_filename = f"{output_dir}/evaluation_report.txt"
    with open(report_filename, 'w') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))
    print(f"\nDetailed report saved to: {report_filename}")

    return report


def evaluate_ads(input_file, output_file=None):
    """
    Main function to evaluate ads from Excel file
    """
    logger.info(f"Starting ad evaluation from file: {input_file}")
    start_time = time.time()

    try:
        # Load the Excel file
        df = pd.read_excel(input_file)
        logger.info(f"Loaded dataframe with {len(df)} rows")
        print(f"Loaded {len(df)} ads from {input_file}")

        # Print column information for debugging
        print(f"\nColumns in the file ({len(df.columns)} total):")
        for i, col in enumerate(df.columns):
            col_value = df[col].iloc[0] if len(df) > 0 else 'N/A'
            print(f"  [{i}] '{col}': {str(col_value)[:100]}...")

        # Evaluate ads using multiprocessing
        print(f"\nStarting ad evaluation with LLM...")
        results = run_multiprocessing(df)

        if not results:
            print("No results generated. Check the logs for errors.")
            logger.error("No results generated from multiprocessing")
            return pd.DataFrame(), None

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Extract scores for easier analysis
        scores = []
        for idx, row in results_df.iterrows():
            if 'evaluation' in row and isinstance(row['evaluation'], dict):
                eval_data = row['evaluation']
                if 'scores' in eval_data:
                    scores.append({
                        'index': row['index'],
                        'headline': row.get('headline', ''),
                        'description': row.get('description', ''),
                        'group': row.get('group', 'Unknown'),
                        'total_score': eval_data['scores'].get('total_score'),
                        'grade': eval_data.get('grade'),
                        'search_intent': eval_data['scores'].get('search_intent_alignment'),
                        'copy_quality': eval_data['scores'].get('persuasive_copy_quality'),
                        'cta': eval_data['scores'].get('cta_effectiveness'),
                        'trust': eval_data['scores'].get('professionalism_trust'),
                        'overall': eval_data['scores'].get('overall_potential')
                    })
                else:
                    logger.warning(f"No scores in evaluation for row {idx}")
            else:
                logger.warning(f"Invalid evaluation data for row {idx}")

        if not scores:
            print("Warning: No valid scores were extracted from evaluations")
            logger.error("No valid scores extracted")
            scores_df = pd.DataFrame()
        else:
            scores_df = pd.DataFrame(scores)

        # Perform statistical analysis if we have scores
        t_test_results = None
        if not scores_df.empty and 'total_score' in scores_df.columns:
            t_test_results = perform_t_test(results_df)
        else:
            print("\nSkipping statistical analysis - no valid scores available")

        # Generate summary report
        report = generate_summary_report(scores_df if not scores_df.empty else results_df, t_test_results)

        # Save results
        if output_file:
            if not scores_df.empty:
                scores_df.to_excel(output_file, index=False)
                print(f"\nResults saved to: {output_file}")
            else:
                print("\nWarning: No scores extracted, saving raw results instead")
                results_df.to_excel(output_file, index=False)

        # Save detailed results to JSON
        json_filename = f"{output_dir}/detailed_results.json"
        results_df.to_json(json_filename, orient='records', indent=2)
        print(f"Detailed results saved to: {json_filename}")

        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 60}")
        print(f"⏰ Total time: {total_time:.2f} seconds")
        print(f"📊 Total ads evaluated: {len(results)}")
        print(f"📁 Results saved in: {output_dir}/")

        return scores_df if not scores_df.empty else results_df, t_test_results

    except Exception as e:
        logger.error(f"Error in evaluate_ads: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    # Input file path - adjust as needed
    input_file = r"F:\CUHK\Course\DOTE6635\replication\web appendix tables and figures\data\20231016-education-prolific-SEAconversionrates-data.xlsx"

    # Output file path
    output_file = "ad_evaluation_results.xlsx"

    try:
        print(f"Starting ad evaluation process...")
        print(f"Input file: {input_file}")
        print(f"Output will be saved to: {output_file}")
        print(f"Temporary files in: {output_dir}/")
        print("=" * 60)

        results_df, stats_results = evaluate_ads(input_file, output_file)

        logger.info("Script completed successfully")
        print("\nScript completed successfully!")

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        print(f"Script failed: {e}")