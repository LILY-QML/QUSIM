"""Test vectorized noise generation performance"""

import time
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'helper'))

from modules.noise import NoiseGenerator, NoiseConfiguration

def test_vectorized_performance():
    """Compare vectorized vs regular noise generation"""
    
    print("🚀 VECTORIZED NOISE PERFORMANCE TEST")
    print("=" * 50)
    
    # Configure noise generator
    config = NoiseConfiguration()
    config.seed = 42
    config.enable_c13_bath = False  # Disable for pure speed test
    config.enable_temperature = False
    config.enable_strain = False
    config.enable_microwave = False
    config.enable_optical = False
    config.enable_charge_noise = False
    
    generator = NoiseGenerator(config)
    
    # Test different sample sizes
    test_sizes = [100, 1000, 10000]
    
    for n_samples in test_sizes:
        print(f"\n📊 Testing {n_samples} samples:")
        
        # Test regular method
        start = time.time()
        regular_noise = generator.get_total_magnetic_noise(n_samples)
        regular_time = time.time() - start
        regular_rate = n_samples / regular_time
        
        # Test vectorized method
        start = time.time()
        vectorized_noise = generator.get_total_magnetic_noise_vectorized(n_samples)
        vectorized_time = time.time() - start
        vectorized_rate = n_samples / vectorized_time
        
        # Calculate speedup
        speedup = vectorized_rate / regular_rate
        
        print(f"  Regular method:    {regular_time*1000:.1f} ms ({regular_rate:.0f} samples/sec)")
        print(f"  Vectorized method: {vectorized_time*1000:.1f} ms ({vectorized_rate:.0f} samples/sec)")
        print(f"  Speedup:           {speedup:.1f}x")
        
        # Verify results are similar (statistically)
        regular_rms = np.sqrt(np.mean(np.sum(regular_noise**2, axis=1)))
        vectorized_rms = np.sqrt(np.mean(np.sum(vectorized_noise**2, axis=1)))
        print(f"  Regular RMS:       {regular_rms*1e12:.1f} pT")
        print(f"  Vectorized RMS:    {vectorized_rms*1e12:.1f} pT")
        
        # Check if results are statistically similar
        relative_diff = abs(regular_rms - vectorized_rms) / regular_rms
        if relative_diff < 0.1:  # Within 10%
            print(f"  ✅ Results match (diff: {relative_diff*100:.1f}%)")
        else:
            print(f"  ⚠️  Results differ by {relative_diff*100:.1f}%")

def test_with_c13():
    """Test performance with C13 bath enabled"""
    
    print("\n\n🧲 PERFORMANCE WITH C13 BATH")
    print("=" * 40)
    
    config = NoiseConfiguration()
    config.seed = 42
    config.enable_temperature = False
    config.enable_strain = False
    config.enable_microwave = False
    config.enable_optical = False
    config.enable_charge_noise = False
    
    # Optimize C13 for speed
    config.parameter_overrides = {
        'c13_bath': {
            'concentration': 0.011,
            'max_distance': 3e-9,
            'cluster_size': 4
        }
    }
    
    generator = NoiseGenerator(config)
    
    # Test performance
    n_samples = 1000
    
    start = time.time()
    noise = generator.get_total_magnetic_noise_vectorized(n_samples)
    total_time = time.time() - start
    rate = n_samples / total_time
    
    print(f"  Sample size:       {n_samples}")
    print(f"  Total time:        {total_time*1000:.1f} ms")
    print(f"  Generation rate:   {rate:.0f} samples/sec")
    
    if rate > 10000:
        print(f"  ✅ TARGET ACHIEVED: >10,000 samples/sec!")
    else:
        print(f"  ⚠️  Below target (need {10000/rate:.1f}x improvement)")

if __name__ == "__main__":
    test_vectorized_performance()
    test_with_c13()
    
    print("\n🏁 PERFORMANCE TEST COMPLETED")