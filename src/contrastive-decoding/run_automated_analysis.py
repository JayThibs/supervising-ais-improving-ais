import os
import argparse
from dotenv import load_dotenv
from automated_divergence_analyzer import AutomatedDivergenceAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Run automated divergence analysis on input file.')
    parser.add_argument('--subtopics', nargs='+', required=True, help='List of subtopics to focus on')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations to run')
    parser.add_argument('--openai_api_key', type=str, help='OpenAI API key')

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Set the OpenAI API key
    openai_api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key not provided. Please set it as an environment variable or pass it as an argument.")
    os.environ['OPENAI_API_KEY'] = openai_api_key

    analyzer = AutomatedDivergenceAnalyzer(
        subtopics=args.subtopics,
        input_file_path=args.input_file
    )

    analyzer.run_analysis_loop(args.num_iterations)

if __name__ == '__main__':
    main()