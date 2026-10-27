from automated_divergence_analyzer import AutomatedDivergenceAnalyzer
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Run automated divergence analysis on input file.')
    parser.add_argument('--subtopics', nargs='+', required=True, help='List of subtopics to focus on')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations to run')
    parser.add_argument('--openai_api_key', type=str, help='OpenAI API key')

    args = parser.parse_args()

    # Set the OpenAI API key
    # The key comes from OPENAI_API_KEY in the environment (removed a hardcoded key before publishing).

    analyzer = AutomatedDivergenceAnalyzer(
        subtopics=args.subtopics,
        input_file_path=args.input_file
    )

    analyzer.run_analysis_loop(args.num_iterations)

if __name__ == '__main__':
    main()