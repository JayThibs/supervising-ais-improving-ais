"""
Basic Example: Mechanistic Discovery Analysis

This script demonstrates how to use the mechanistic discovery module to
compare two models and find behavioral differences efficiently.

Prerequisites:
1. Install circuit-tracer: pip install circuit-tracer
2. Have two models to compare (e.g., base and fine-tuned)
3. Have pre-trained transcoders for the models

Example usage:
    python basic_analysis.py --base-model gpt2 --intervention-model gpt2-finetuned
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mechanistic_discovery import (
    MechanisticBehavioralAnalyzer, 
    AnalysisConfig,
    CircuitVisualizer
)
from mechanistic_discovery.utils import check_model_compatibility


def main():
    """
    Main function demonstrating basic mechanistic discovery workflow.
    """
    parser = argparse.ArgumentParser(
        description="Compare two models using mechanistic discovery"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Name or path of the base model"
    )
    parser.add_argument(
        "--intervention-model", 
        type=str,
        required=True,
        help="Name or path of the intervention model"
    )
    parser.add_argument(
        "--transcoder-set",
        type=str,
        default=None,
        help="Path to transcoder set (defaults to model-name-transcoders)"
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=50,
        help="Number of seed prompts to analyze"
    )
    parser.add_argument(
        "--focus-areas",
        nargs="+",
        default=["safety", "capabilities", "reasoning"],
        help="Areas to focus analysis on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mechanistic_analysis_results",
        help="Directory for results"
    )
    
    args = parser.parse_args()
    
    # Step 1: Check model compatibility
    print("=" * 80)
    print("MECHANISTIC DISCOVERY ANALYSIS")
    print("=" * 80)
    
    print(f"\nChecking model compatibility...")
    print(f"Base model: {args.base_model}")
    print(f"Intervention model: {args.intervention_model}")
    
    is_compatible, issues = check_model_compatibility(
        args.base_model,
        args.intervention_model
    )
    
    if not is_compatible:
        print("\n❌ Models are not compatible for comparison:")
        for issue in issues:
            print(f"   - {issue}")
        return
    
    if issues:
        print("\n⚠️  Compatibility warnings:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Models are fully compatible")
    
    # Step 2: Configure analysis
    print("\nConfiguring analysis...")
    
    # Default transcoder set name
    if args.transcoder_set is None:
        # Extract model name for transcoder path
        base_model_name = args.base_model.split('/')[-1]
        args.transcoder_set = f"{base_model_name}-transcoders"
        print(f"Using default transcoder set: {args.transcoder_set}")
    
    config = AnalysisConfig(
        base_model_name=args.base_model,
        intervention_model_name=args.intervention_model,
        transcoder_set=args.transcoder_set,
        n_seed_prompts=args.n_prompts,
        n_test_prompts_per_hypothesis=50,
        max_hypotheses_to_test=10,
        focus_areas=args.focus_areas,
        output_dir=args.output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=4 if torch.cuda.is_available() else 2
    )
    
    print(f"Focus areas: {', '.join(args.focus_areas)}")
    print(f"Device: {config.device}")
    print(f"Output directory: {config.output_dir}")
    
    # Step 3: Initialize analyzer
    print("\nInitializing analyzer...")
    try:
        analyzer = MechanisticBehavioralAnalyzer(config)
    except Exception as e:
        print(f"\n❌ Failed to initialize analyzer: {e}")
        return
    
    # Step 4: Generate seed prompts
    print(f"\nGenerating {args.n_prompts} seed prompts...")
    
    # You can provide custom prompts or use generated ones
    seed_prompts = None  # Will use auto-generated prompts
    
    # Step 5: Run analysis
    print("\nRunning analysis (this may take 1-2 hours)...")
    print("Phases:")
    print("  1. Circuit analysis")
    print("  2. Pattern recognition") 
    print("  3. Hypothesis generation")
    print("  4. Behavioral validation")
    print("  5. Active exploration (if enabled)")
    
    try:
        report = analyzer.analyze(seed_prompts=seed_prompts)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Visualize results
    print("\nGenerating visualizations...")
    
    visualizer = CircuitVisualizer(output_dir=f"{args.output_dir}/visualizations")
    
    # Create summary dashboard
    try:
        dashboard = visualizer.create_summary_dashboard(report)
        print(f"✅ Dashboard saved to {args.output_dir}/visualizations/analysis_dashboard.html")
    except Exception as e:
        print(f"⚠️  Could not create dashboard: {e}")
    
    # Visualize top circuit differences
    if report.circuit_differences:
        print("\nVisualizing top circuit differences...")
        for i, circuit_diff in enumerate(report.circuit_differences[:3]):
            try:
                fig, _ = visualizer.visualize_circuit_difference(
                    circuit_diff,
                    save_name=f"circuit_diff_{i}.png"
                )
                plt.close(fig)
            except Exception as e:
                print(f"⚠️  Could not visualize circuit {i}: {e}")
    
    # Step 7: Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Validated behavioral changes
    validated = [r for r in report.validation_results if r.is_validated()]
    
    if validated:
        print(f"\n✅ Found {len(validated)} validated behavioral changes:")
        
        # Group by hypothesis type
        by_type = {}
        for result in validated:
            h_type = result.hypothesis.hypothesis_type.value
            if h_type not in by_type:
                by_type[h_type] = []
            by_type[h_type].append(result)
        
        for h_type, results in by_type.items():
            print(f"\n{h_type.upper()} ({len(results)} findings):")
            
            # Show top 3 by effect size
            top_results = sorted(results, key=lambda r: r.effect_size, reverse=True)[:3]
            
            for i, result in enumerate(top_results, 1):
                print(f"\n  {i}. {result.hypothesis.description}")
                print(f"     Effect size: {result.effect_size:.3f}")
                print(f"     P-value: {result.p_value:.4f}")
                print(f"     Confidence: {result.confidence:.2f}")
                
                # Show example
                if result.test_results:
                    example = next(
                        (r for r in result.test_results if r.responses_differ()),
                        None
                    )
                    if example:
                        print(f"\n     Example:")
                        print(f"     Prompt: {example.test_case.prompt[:80]}...")
                        print(f"     Base: {example.base_response[:100]}...")
                        print(f"     Intervention: {example.intervention_response[:100]}...")
    else:
        print("\n❌ No statistically significant behavioral changes detected")
        print("   This could mean:")
        print("   - The models behave very similarly")
        print("   - More prompts are needed for detection")
        print("   - The changes are too subtle for current methods")
    
    # Circuit-level insights
    print(f"\n📊 Circuit Analysis Summary:")
    if report.systematic_patterns:
        n_systematic = report.systematic_patterns.get('n_systematic_features', 0)
        print(f"   - Systematically changing features: {n_systematic}")
        
        if 'consistent_features' in report.systematic_patterns:
            # Show top systematic features
            top_features = list(report.systematic_patterns['consistent_features'].items())[:5]
            if top_features:
                print("\n   Top changing features:")
                for feat_key, info in top_features:
                    layer = info['layer']
                    idx = info['feature_idx']
                    freq = info['frequency']
                    print(f"     - Layer {layer}, Feature {idx}: {freq:.1%} of prompts")
    
    # Performance metrics
    print(f"\n⏱️  Performance Metrics:")
    total_time = report.runtime_info.get('total_runtime', 0)
    print(f"   Total runtime: {total_time/60:.1f} minutes")
    
    if total_time > 0:
        breakdown = []
        for phase, time in report.runtime_info.items():
            if phase != 'total_runtime' and time > 0:
                percentage = (time / total_time) * 100
                breakdown.append((phase, time, percentage))
        
        breakdown.sort(key=lambda x: x[1], reverse=True)
        
        print("\n   Time breakdown:")
        for phase, time, pct in breakdown[:5]:
            print(f"     - {phase}: {time/60:.1f} min ({pct:.1f}%)")
    
    # Save path
    print(f"\n📁 Full report saved to: {args.output_dir}/analysis_report_{report.timestamp}.json")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    

if __name__ == "__main__":
    # Check for required imports
    try:
        import torch
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install: pip install torch matplotlib")
        sys.exit(1)
    
    main()