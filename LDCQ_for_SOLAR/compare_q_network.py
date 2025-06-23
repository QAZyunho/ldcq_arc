import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

def load_and_compare_models(model1_path, model2_path):
    """두 개의 DQN 모델을 불러와서 파라미터를 비교하는 함수"""
    
    # 첫 번째 모델 로드
    dqn_agent1 = torch.load(model1_path).to('cuda')
    
    # 두 번째 모델 로드
    dqn_agent2 = torch.load(model2_path).to('cuda')
    
    print("=" * 60)
    print("모델 파라미터 비교")
    print("=" * 60)
    
    # 1. 기본 정보 비교
    print(f"모델 1 (steps: {model1_path}):")
    print(f"  - 타입: {type(dqn_agent1)}")
    print(f"  - 디바이스: {next(dqn_agent1.parameters()).device}")
    
    print(f"\n모델 2 (steps: {model2_path}):")
    print(f"  - 타입: {type(dqn_agent2)}")
    print(f"  - 디바이스: {next(dqn_agent2.parameters()).device}")
    
    # 2. 파라미터 개수 비교
    params1 = sum(p.numel() for p in dqn_agent1.parameters())
    params2 = sum(p.numel() for p in dqn_agent2.parameters())
    
    print(f"\n파라미터 개수:")
    print(f"  - 모델 1: {params1:,}")
    print(f"  - 모델 2: {params2:,}")
    print(f"  - 차이: {abs(params1 - params2):,}")
    
    # 3. 레이어별 파라미터 차원 비교
    print(f"\n레이어별 파라미터 차원 비교:")
    print("-" * 60)
    
    state_dict1 = dqn_agent1.state_dict()
    state_dict2 = dqn_agent2.state_dict()
    
    # 공통 레이어 찾기
    common_keys = set(state_dict1.keys()) & set(state_dict2.keys())
    model1_only = set(state_dict1.keys()) - set(state_dict2.keys())
    model2_only = set(state_dict2.keys()) - set(state_dict1.keys())
    
    print(f"공통 레이어 수: {len(common_keys)}")
    print(f"모델1에만 있는 레이어: {len(model1_only)}")
    print(f"모델2에만 있는 레이어: {len(model2_only)}")
    
    # 공통 레이어 차원 비교
    dimension_differences = []
    
    for key in sorted(common_keys):
        shape1 = state_dict1[key].shape
        shape2 = state_dict2[key].shape
        
        if shape1 != shape2:
            dimension_differences.append((key, shape1, shape2))
            print(f"❌ {key}:")
            print(f"    모델1: {shape1}")
            print(f"    모델2: {shape2}")
        else:
            print(f"✅ {key}: {shape1}")
    
    # 4. 차원이 다른 레이어 요약
    if dimension_differences:
        print(f"\n차원이 다른 레이어 요약:")
        print("-" * 40)
        for key, shape1, shape2 in dimension_differences:
            print(f"{key}: {shape1} vs {shape2}")
    else:
        print(f"\n✅ 모든 공통 레이어의 차원이 동일합니다!")
    
    # 5. 모델1에만 있는 레이어
    if model1_only:
        print(f"\n모델1에만 있는 레이어:")
        for key in sorted(model1_only):
            print(f"  - {key}: {state_dict1[key].shape}")
    
    # 6. 모델2에만 있는 레이어  
    if model2_only:
        print(f"\n모델2에만 있는 레이어:")
        for key in sorted(model2_only):
            print(f"  - {key}: {state_dict2[key].shape}")
    
    return dqn_agent1, dqn_agent2, dimension_differences

def compare_specific_layers(model1, model2, layer_names):
    """특정 레이어들의 파라미터를 자세히 비교"""
    
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    print("\n특정 레이어 상세 비교:")
    print("-" * 40)
    
    for layer_name in layer_names:
        if layer_name in state_dict1 and layer_name in state_dict2:
            param1 = state_dict1[layer_name]
            param2 = state_dict2[layer_name]
            
            print(f"\n레이어: {layer_name}")
            print(f"  형태: {param1.shape}")
            print(f"  데이터 타입: {param1.dtype}")
            print(f"  파라미터 개수: {param1.numel()}")
            
            # 파라미터 값 차이 계산
            if param1.shape == param2.shape:
                diff = torch.abs(param1 - param2)
                print(f"  최대 차이: {diff.max().item():.6f}")
                print(f"  평균 차이: {diff.mean().item():.6f}")
                print(f"  차이 표준편차: {diff.std().item():.6f}")
            else:
                print(f"  ❌ 형태가 다름: {param1.shape} vs {param2.shape}")

def compare_all_weights(model1, model2, detailed=True, plot_histogram=False):
    """모든 레이어의 weight 값 차이를 비교하는 함수"""
    
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    print("\n" + "=" * 60)
    print("전체 Weight 값 차이 분석")
    print("=" * 60)
    
    all_diffs = []
    layer_stats = []
    
    for key in sorted(state_dict1.keys()):
        if key in state_dict2:
            param1 = state_dict1[key]
            param2 = state_dict2[key]
            
            if param1.shape == param2.shape:
                diff = torch.abs(param1 - param2)
                all_diffs.append(diff.flatten())
                
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                std_diff = diff.std().item()
                median_diff = diff.median().item()
                
                # 차이가 거의 없는 파라미터 비율
                zero_diff_ratio = (diff < 1e-8).float().mean().item()
                
                layer_stats.append({
                    'name': key,
                    'max': max_diff,
                    'mean': mean_diff,
                    'std': std_diff,
                    'median': median_diff,
                    'zero_ratio': zero_diff_ratio,
                    'param_count': param1.numel()
                })
                
                if detailed:
                    print(f"\n📊 {key}:")
                    print(f"   형태: {param1.shape}")
                    print(f"   최대 차이: {max_diff:.8f}")
                    print(f"   평균 차이: {mean_diff:.8f}")
                    print(f"   중앙값 차이: {median_diff:.8f}")
                    print(f"   표준편차: {std_diff:.8f}")
                    print(f"   동일한 값 비율: {zero_diff_ratio*100:.2f}%")
    
    # 전체 통계
    if all_diffs:
        all_diffs_tensor = torch.cat(all_diffs)
        
        print(f"\n🔍 전체 모델 Weight 차이 통계:")
        print(f"   총 파라미터 수: {len(all_diffs_tensor):,}")
        print(f"   최대 차이: {all_diffs_tensor.max().item():.8f}")
        print(f"   평균 차이: {all_diffs_tensor.mean().item():.8f}")
        print(f"   중앙값 차이: {all_diffs_tensor.median().item():.8f}")
        print(f"   표준편차: {all_diffs_tensor.std().item():.8f}")
        print(f"   동일한 값 비율: {(all_diffs_tensor < 1e-8).float().mean().item()*100:.2f}%")
        
        # 차이 분포 분석
        print(f"\n📈 차이 분포 분석:")
        percentiles = [50, 75, 90, 95, 99, 99.9]
        for p in percentiles:
            val = torch.quantile(all_diffs_tensor, p/100.0).item()
            print(f"   {p}% 백분위수: {val:.8f}")
    
    # 가장 차이가 큰 레이어들
    print(f"\n🔥 차이가 가장 큰 레이어 TOP 5:")
    layer_stats.sort(key=lambda x: x['max'], reverse=True)
    for i, stat in enumerate(layer_stats[:5]):
        print(f"   {i+1}. {stat['name']}: 최대차이 {stat['max']:.8f}, 평균차이 {stat['mean']:.8f}")
    
    # 히스토그램 플롯
    if plot_histogram and all_diffs:
        plt.figure(figsize=(12, 8))
        
        # 로그 스케일로 히스토그램
        plt.subplot(2, 2, 1)
        log_diffs = torch.log10(all_diffs_tensor + 1e-10).cpu().numpy()
        plt.hist(log_diffs, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Log10(Weight Difference)')
        plt.ylabel('Frequency')
        plt.title('Weight Differences Distribution (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        # 선형 스케일 (작은 차이만)
        plt.subplot(2, 2, 2)
        small_diffs = all_diffs_tensor[all_diffs_tensor < 0.1].cpu().numpy()
        plt.hist(small_diffs, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Weight Difference')
        plt.ylabel('Frequency')
        plt.title('Small Weight Differences (<0.1)')
        plt.grid(True, alpha=0.3)
        
        # 레이어별 최대 차이
        plt.subplot(2, 2, 3)
        layer_names = [stat['name'].split('.')[-1] for stat in layer_stats]
        max_diffs = [stat['max'] for stat in layer_stats]
        plt.bar(range(len(layer_names)), max_diffs)
        plt.xlabel('Layer Index')
        plt.ylabel('Max Difference')
        plt.title('Max Difference by Layer')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 레이어별 평균 차이
        plt.subplot(2, 2, 4)
        mean_diffs = [stat['mean'] for stat in layer_stats]
        plt.bar(range(len(layer_names)), mean_diffs)
        plt.xlabel('Layer Index')
        plt.ylabel('Mean Difference')
        plt.title('Mean Difference by Layer')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return layer_stats, all_diffs_tensor if all_diffs else None

def find_similar_layers(model1, model2, threshold=1e-6):
    """거의 동일한 weight를 가진 레이어들을 찾는 함수"""
    
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    similar_layers = []
    different_layers = []
    
    for key in sorted(state_dict1.keys()):
        if key in state_dict2:
            param1 = state_dict1[key]
            param2 = state_dict2[key]
            
            if param1.shape == param2.shape:
                diff = torch.abs(param1 - param2)
                max_diff = diff.max().item()
                
                if max_diff < threshold:
                    similar_layers.append((key, max_diff))
                else:
                    different_layers.append((key, max_diff))
    
    print(f"\n🎯 임계값 {threshold}으로 레이어 분류:")
    print(f"   거의 동일한 레이어: {len(similar_layers)}개")
    print(f"   차이가 있는 레이어: {len(different_layers)}개")
    
    if similar_layers:
        print(f"\n✅ 거의 동일한 레이어들:")
        for name, diff in similar_layers[:10]:  # 상위 10개만
            print(f"   {name}: 최대차이 {diff:.2e}")
    
    if different_layers:
        print(f"\n❌ 차이가 있는 레이어들:")
        for name, diff in sorted(different_layers, key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {name}: 최대차이 {diff:.2e}")
    
    return similar_layers, different_layers

def weight_change_summary(model1, model2):
    """Weight 변화 요약 리포트"""
    
    print("\n" + "🎯" * 20)
    print("Weight 변화 요약 리포트")
    print("🎯" * 20)
    
    # 전체 분석
    layer_stats, all_diffs = compare_all_weights(model1, model2, detailed=False)
    
    if all_diffs is not None:
        # 변화 정도 분류
        no_change = (all_diffs < 1e-8).sum().item()
        tiny_change = ((all_diffs >= 1e-8) & (all_diffs < 1e-4)).sum().item()
        small_change = ((all_diffs >= 1e-4) & (all_diffs < 1e-2)).sum().item()
        medium_change = ((all_diffs >= 1e-2) & (all_diffs < 1e-1)).sum().item()
        large_change = (all_diffs >= 1e-1).sum().item()
        total = len(all_diffs)
        
        print(f"\n📊 파라미터 변화 분포:")
        print(f"   변화 없음 (<1e-8):     {no_change:8,} ({no_change/total*100:5.1f}%)")
        print(f"   미세한 변화 (1e-8~1e-4): {tiny_change:8,} ({tiny_change/total*100:5.1f}%)")
        print(f"   작은 변화 (1e-4~1e-2):   {small_change:8,} ({small_change/total*100:5.1f}%)")
        print(f"   중간 변화 (1e-2~1e-1):   {medium_change:8,} ({medium_change/total*100:5.1f}%)")
        print(f"   큰 변화 (>1e-1):        {large_change:8,} ({large_change/total*100:5.1f}%)")
        
        # 모델이 얼마나 다른지 판단
        changed_ratio = (all_diffs >= 1e-6).float().mean().item()
        if changed_ratio < 0.01:
            print(f"\n🔹 결론: 모델들이 거의 동일합니다. (변화된 파라미터 < 1%)")
        elif changed_ratio < 0.1:
            print(f"\n🔸 결론: 모델들이 매우 유사합니다. (변화된 파라미터 {changed_ratio*100:.1f}%)")
        elif changed_ratio < 0.5:
            print(f"\n🔶 결론: 모델들이 어느 정도 다릅니다. (변화된 파라미터 {changed_ratio*100:.1f}%)")
        else:
            print(f"\n🔴 결론: 모델들이 상당히 다릅니다. (변화된 파라미터 {changed_ratio*100:.1f}%)")

# 사용 예시
if __name__ == "__main__":
    # 기존 모델 경로들
    model1_path = "/home/jovyan/beomi/jaehyun/ldcq_arc/LDCQ_for_SOLAR/q_checkpoints/gpu2_09.13_0.5_100/Openhpc_gpu2_ARCLE_09.13_400__dqn_agent_150_cfg_weight_0.0_PERbuffer.pt"
    model2_path = "/home/jovyan/beomi/jaehyun/ldcq_arc/LDCQ_for_SOLAR/q_checkpoints/gpu2_09.13_0.5_500/Openhpc_gpu2_ARCLE_09.13_400__dqn_agent_150_cfg_weight_0.0_PERbuffer.pt"
    
    # 1. 기본 구조 및 차원 비교
    print("1️⃣ 기본 구조 및 차원 비교")
    model1, model2, differences = load_and_compare_models(model1_path, model2_path)
    
    # 2. Weight 값 차이 분석
    print("\n" + "="*60)
    print("2️⃣ Weight 값 차이 분석 시작")
    print("="*60)
    
    # 전체 weight 차이 분석 (상세 모드)
    layer_stats, all_diffs = compare_all_weights(model1, model2, detailed=True)
    
    # 3. 거의 동일한 레이어 찾기
    print("\n3️⃣ 유사 레이어 분석")
    similar_layers, different_layers = find_similar_layers(model1, model2, threshold=1e-6)
    
    # 4. 요약 리포트
    print("\n4️⃣ 최종 요약 리포트")
    weight_change_summary(model1, model2)
    
    # 5. 특정 레이어 비교 (선택사항 - 필요시 주석 해제)
    # print("\n5️⃣ 특정 레이어 상세 분석")
    # compare_specific_layers(model1, model2, ['q_network.fc1.weight', 'q_network.fc2.weight'])
    
    print("\n🎉 분석 완료!")