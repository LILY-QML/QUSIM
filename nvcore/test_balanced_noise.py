"""Test balanced noise configuration"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'helper'))

from modules.noise import NoiseGenerator, NoiseConfiguration

def test_balanced_noise():
    """Test the balanced_lab preset for proper noise ratios"""
    
    print("🎯 BALANCED NOISE CONFIGURATION TEST")
    print("=" * 50)
    
    # Test different presets
    presets = ['room_temperature', 'balanced_lab']
    
    for preset_name in presets:
        print(f"\n📊 Testing preset: {preset_name}")
        print("-" * 40)
        
        # Create configuration
        config = NoiseConfiguration.from_preset(preset_name)
        config.seed = 42
        
        # Optimize C13 for speed
        if 'c13_bath' not in config.parameter_overrides:
            config.parameter_overrides['c13_bath'] = {}
        config.parameter_overrides['c13_bath']['max_distance'] = 3e-9
        config.parameter_overrides['c13_bath']['cluster_size'] = 4
        
        # Disable slow sources
        config.enable_temperature = False
        config.enable_strain = False
        config.enable_microwave = False
        config.enable_optical = False
        config.enable_charge_noise = False
        
        # Create generator
        generator = NoiseGenerator(config)
        
        # Test individual sources
        source_contributions = {}
        
        for source_name, source in generator.sources.items():
            # Sample from each source
            samples = source.sample(100)
            
            if samples.ndim > 1 and samples.shape[1] == 3:
                rms = np.sqrt(np.mean(np.sum(samples**2, axis=1)))
            else:
                rms = np.sqrt(np.mean(samples**2))
            
            source_contributions[source_name] = rms
            print(f"  {source_name:15}: {rms*1e12:8.1f} pT")
        
        # Calculate total noise
        total_samples = generator.get_total_magnetic_noise_vectorized(100)
        total_rms = np.sqrt(np.mean(np.sum(total_samples**2, axis=1)))
        print(f"  {'TOTAL':15}: {total_rms*1e12:8.1f} pT")
        
        # Calculate ratios
        if 'c13_bath' in source_contributions and 'external_field' in source_contributions:
            c13_rms = source_contributions['c13_bath']
            ext_rms = source_contributions['external_field']
            
            if ext_rms > 0:
                ratio = c13_rms / ext_rms
                print(f"\n  C13/External ratio: {ratio:.3f}")
                
                if preset_name == 'balanced_lab':
                    if 0.1 <= ratio <= 10:
                        print("  ✅ BALANCED: Ratio in target range [0.1, 10]")
                    else:
                        print(f"  ⚠️  UNBALANCED: Ratio {ratio:.3f} outside [0.1, 10]")
        
        # Component breakdown
        print("\n  Component Breakdown:")
        for source_name, rms in source_contributions.items():
            percentage = (rms / total_rms) * 100 if total_rms > 0 else 0
            print(f"    {source_name:12}: {percentage:5.1f}%")

def test_enhancement_scaling():
    """Test that field enhancement properly scales C13 contribution"""
    
    print("\n\n🔬 C13 FIELD ENHANCEMENT TEST")
    print("=" * 40)
    
    base_config = NoiseConfiguration()
    base_config.seed = 42
    base_config.enable_external_field = False
    base_config.enable_johnson = False
    base_config.enable_temperature = False
    base_config.enable_strain = False
    base_config.enable_microwave = False
    base_config.enable_optical = False
    base_config.enable_charge_noise = False
    
    # Test different enhancement factors
    enhancement_factors = [1.0, 2.0, 5.0]
    
    for factor in enhancement_factors:
        print(f"\n  Enhancement factor: {factor:.1f}x")
        
        config = NoiseConfiguration.from_dict(base_config.__dict__)
        config.parameter_overrides = {
            'c13_bath': {
                'concentration': 0.011,
                'max_distance': 3e-9,
                'cluster_size': 4,
                'field_enhancement_factor': factor
            }
        }
        
        generator = NoiseGenerator(config)
        
        # Sample C13 noise
        samples = generator.sources['c13_bath'].sample(50)
        rms = np.sqrt(np.mean(np.sum(samples**2, axis=1)))
        
        print(f"    C13 RMS: {rms*1e12:.1f} pT")
        
        if factor == 1.0:
            baseline_rms = rms
        else:
            actual_scaling = rms / baseline_rms
            print(f"    Actual scaling: {actual_scaling:.2f}x (expected {factor:.1f}x)")
            
            if abs(actual_scaling - factor) / factor < 0.1:
                print("    ✅ Enhancement working correctly")
            else:
                print("    ⚠️  Enhancement not scaling properly")

if __name__ == "__main__":
    test_balanced_noise()
    test_enhancement_scaling()
    
    print("\n🏁 BALANCED NOISE TEST COMPLETED")