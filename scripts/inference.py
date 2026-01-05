"""
Brain Tokenizer Inference Script
Input: (W, C, T) EEG data - W: windows, C: channels (62), T: timepoints (400)
Output: Encoded features per window
"""

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
import sys
sys.path.insert(0, '/share/project/BrainToken')
from braintoken import get_brvq_model


def load_data(eeg_path, coord_path, sample_idx=0):
    """Load EEG data and electrode coordinates"""
    # Load electrode coordinates (62, 3)
    coord_data = np.load(coord_path)
    coord = torch.from_numpy(coord_data.astype(np.float32)).cuda()
    
    # Load EEG dataset (7, 40, 5, 62, 400) -> (1400, 62, 400)
    full_dataset = np.load(eeg_path)
    full_dataset = full_dataset.reshape(-1, 62, 400)
    
    # Get single sample (62, 400)
    eeg_sample = full_dataset[sample_idx]
    eeg_tensor = torch.from_numpy(eeg_sample.astype(np.float32)).unsqueeze(0).cuda()  # (1, 62, 400)
    coord = coord.unsqueeze(0)  # (1, 62, 3)
    
    return eeg_tensor, coord, full_dataset


def load_model(checkpoint_path, device='cuda'):
    """Load brain tokenizer model"""
    brain_tokenizer = get_brvq_model('stageI', device=device)
    checkpoint = torch.load(checkpoint_path, weights_only=False)['state_dict']
    brain_tokenizer.load_state_dict(checkpoint, strict=False)
    brain_tokenizer.eval()
    return brain_tokenizer


def preprocess_data(eeg_tensor, coord, target_time=480, num_windows=2):
    """
    Preprocess EEG data for inference
    (1, 62, 400) -> (num_windows, 62, 480) -> (num_windows, 62, 240)
    Windows: temporal segments of the EEG signal
    """
    # Time upsampling: 400 -> target_time
    eeg_upsampled = F.interpolate(
        eeg_tensor.unsqueeze(2),
        size=(1, target_time),
        mode='bilinear',
        align_corners=False
    ).squeeze(2)
    
    # Windows expansion: direct replication
    eeg_windows = eeg_upsampled.repeat(num_windows, 1, 1)
    coord_windows = coord.repeat(num_windows, 1, 1)
    
    # Downsample to model expected size: 480 -> 240
    eeg_240 = F.interpolate(
        eeg_windows.unsqueeze(2),
        size=(1, 240),
        mode='bilinear',
        align_corners=False
    ).squeeze(2)
    
    return eeg_240, coord_windows


def inference(brain_tokenizer, eeg_data, coord_data):
    """Run inference and extract features"""
    with torch.no_grad():
        quant, emb_loss, info, pool = brain_tokenizer.encode(eeg_data, coord_data)
    
    # Reshape: (W*C, D, T) -> (W, -1)
    # W: num_windows, C: num_channels
    num_windows = eeg_data.shape[0]
    # quant shape: (124, 4, 30) --> (2, 7440) when num_windows=2, C=62
    quant_per_window = quant.reshape(num_windows, -1)
    
    return {
        'quant': quant,
        'quant_per_window': quant_per_window,
        'pool': pool,
        'emb_loss': emb_loss,
        'info': info
    }


def analyze_token_indices(info_list, num_samples_to_analyze=10, brain_tokenizer=None):
    """
    分析 token indices 的统计信息
    info[2] 是 min_encoding_indices
    
    Args:
        info_list: List of info tuples from inference
        num_samples_to_analyze: Number of samples to analyze in detail
        brain_tokenizer: Optional model to get actual codebook size
    """
    print("\n" + "="*60)
    print("TOKEN INDICES ANALYSIS")
    print("="*60)
    
    # 尝试从模型中获取真实的codebook大小
    actual_codebook_size = None
    codebook_source = None
    if brain_tokenizer is not None:
        try:
            # 尝试从quantizer中获取codebook大小
            quantizer = None
            if hasattr(brain_tokenizer, 'quantize'):
                quantizer = brain_tokenizer.quantize
            elif hasattr(brain_tokenizer, 'module') and hasattr(brain_tokenizer.module, 'quantize'):
                quantizer = brain_tokenizer.module.quantize
            
            if quantizer is not None:
                # 优先使用re_embed（如果有remap，这是实际使用的codebook大小）
                if hasattr(quantizer, 're_embed'):
                    actual_codebook_size = quantizer.re_embed
                    codebook_source = "model.quantize.re_embed (remapped codebook size)"
                # 其次使用n_e（原始codebook大小）
                elif hasattr(quantizer, 'n_e'):
                    actual_codebook_size = quantizer.n_e
                    codebook_source = "model.quantize.n_e (original codebook size)"
                # 最后从embedding层获取
                elif hasattr(quantizer, 'embedding') and hasattr(quantizer.embedding, 'num_embeddings'):
                    actual_codebook_size = quantizer.embedding.num_embeddings
                    codebook_source = "model.quantize.embedding.num_embeddings"
        except Exception as e:
            print(f"  ⚠️  Warning: Could not get codebook size from model: {e}")
    
    if actual_codebook_size is not None:
        print(f"  Codebook size from model: {actual_codebook_size} ({codebook_source})")
    
    # 提取所有样本的 indices
    all_indices = []
    for info in info_list:
        if info is not None and len(info) > 2:
            indices = info[2]  # min_encoding_indices
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu()
            all_indices.append(indices)
    
    if len(all_indices) == 0:
        print("WARNING: No indices found in info_list!")
        return
    
    # 分析前几个样本的详细信息
    print(f"\nAnalyzing first {min(num_samples_to_analyze, len(all_indices))} samples:")
    print("-"*60)
    
    for i in range(min(num_samples_to_analyze, len(all_indices))):
        indices = all_indices[i]
        if indices is None:
            continue
            
        # 展平 indices
        indices_flat = indices.flatten()
        unique_indices = torch.unique(indices_flat)
        unique_count = len(unique_indices)
        total_count = len(indices_flat)
        
        # 统计每个 index 的使用频率
        indices_long = indices_flat.long()
        max_idx = indices_long.max().item() + 1
        index_counts = torch.bincount(indices_long, minlength=max_idx)
        most_common_idx = torch.argmax(index_counts).item()
        most_common_count = index_counts[most_common_idx].item()
        most_common_ratio = most_common_count / total_count
        
        print(f"\nSample {i}:")
        print(f"  Shape: {indices.shape}")
        print(f"  Total tokens: {total_count}")
        print(f"  Unique indices: {unique_count}")
        print(f"  Index range: [{indices_flat.min().item()}, {indices_flat.max().item()}]")
        print(f"  Most common index: {most_common_idx} (used {most_common_count} times, {most_common_ratio*100:.2f}%)")
        
        # 显示前10个最常用的 indices
        top_k = min(10, len(index_counts))
        top_indices = torch.topk(index_counts, top_k)
        print(f"  Top {top_k} most used indices:")
        for j, (idx, count) in enumerate(zip(top_indices.indices, top_indices.values)):
            print(f"    {j+1}. Index {idx.item()}: {count.item()} times ({count.item()/total_count*100:.2f}%)")
    
    # 计算样本间的相似度
    print(f"\n" + "-"*60)
    print("Sample Similarity Analysis:")
    print("-"*60)
    
    num_compare = min(20, len(all_indices))
    similarity_matrix = np.zeros((num_compare, num_compare))
    
    for i in range(num_compare):
        for j in range(i, num_compare):
            if i == j:
                similarity = 1.0
            else:
                idx1 = all_indices[i].flatten()
                idx2 = all_indices[j].flatten()
                
                # 计算相同位置的相同 indices 比例
                if len(idx1) == len(idx2):
                    same = (idx1 == idx2).sum().item()
                    similarity = same / len(idx1)
                else:
                    similarity = 0.0
                
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    # 打印相似度统计
    upper_triangle = similarity_matrix[np.triu_indices(num_compare, k=1)]
    print(f"\nComparing first {num_compare} samples:")
    print(f"  Mean similarity: {upper_triangle.mean():.4f}")
    print(f"  Std similarity: {upper_triangle.std():.4f}")
    print(f"  Min similarity: {upper_triangle.min():.4f}")
    print(f"  Max similarity: {upper_triangle.max():.4f}")
    
    # 找出最相似的样本对
    if num_compare > 1:
        max_sim = upper_triangle.max()
        # 找到最大相似度在原始矩阵中的位置
        max_pos = np.where(similarity_matrix == max_sim)
        if len(max_pos[0]) > 0:
            i_max, j_max = max_pos[0][0], max_pos[1][0]
            print(f"\n  Most similar pair: Sample {i_max} vs Sample {j_max} (similarity: {max_sim:.4f})")
        
        # 检查是否有异常高的相似度
        high_sim_threshold = 0.9
        high_sim_pairs = np.sum(upper_triangle > high_sim_threshold)
        print(f"  Pairs with similarity > {high_sim_threshold}: {high_sim_pairs}")
        if high_sim_pairs > num_compare * 0.1:  # 如果超过10%的样本对相似度很高
            print(f"  ⚠️  WARNING: Many sample pairs have very high similarity!")
            print(f"     This suggests the tokenizer may not be differentiating inputs well.")
    
    # 统计所有样本的 indices 使用情况
    print(f"\n" + "-"*60)
    print("Global Index Usage Statistics:")
    print("-"*60)
    
    all_indices_flat = torch.cat([idx.flatten() for idx in all_indices])
    all_unique = torch.unique(all_indices_flat)
    all_indices_long = all_indices_flat.long()
    max_idx_observed = all_indices_long.max().item()
    min_idx_observed = all_indices_long.min().item()
    
    # 确定codebook大小：优先使用模型中的真实大小，否则使用观察到的最大值+1
    if actual_codebook_size is not None:
        total_codebook_size = actual_codebook_size
        # 使用实际codebook大小进行bincount
        all_counts = torch.bincount(all_indices_long, minlength=total_codebook_size)
    else:
        # 从观察到的indices推断（可能不准确）
        inferred_size = max_idx_observed + 1
        total_codebook_size = inferred_size
        all_counts = torch.bincount(all_indices_long, minlength=total_codebook_size)
        print(f"  ⚠️  WARNING: Codebook size inferred from observed indices: {inferred_size}")
        print(f"      This may be inaccurate. Please ensure model is passed to analyze_token_indices().")
    
    print(f"  Total tokens across all samples: {len(all_indices_flat)}")
    print(f"  Total unique indices used: {len(all_unique)}")
    print(f"  Index range: [{min_idx_observed}, {max_idx_observed}]")
    
    # 统计哪些 indices 从未被使用
    if hasattr(all_counts, 'shape') and len(all_counts) > 0:
        unused_indices = (all_counts == 0).sum().item()
        used_indices = total_codebook_size - unused_indices
        
        print(f"  Codebook size: {total_codebook_size}")
        if actual_codebook_size is not None and codebook_source is not None:
            print(f"    ✓ Retrieved from model: {codebook_source}")
            # 验证观察到的indices是否在有效范围内
            if max_idx_observed >= total_codebook_size:
                print(f"    ⚠️  WARNING: Observed max index ({max_idx_observed}) >= codebook size ({total_codebook_size})!")
            elif max_idx_observed < total_codebook_size * 0.1:
                print(f"    ℹ️  Note: Only using lower {max_idx_observed/total_codebook_size*100:.1f}% of codebook range")
        else:
            print(f"    ⚠️  (Inferred from max observed index: {max_idx_observed} + 1)")
            print(f"    ⚠️  WARNING: Could not retrieve actual codebook size from model!")
            print(f"       Please ensure brain_tokenizer is passed to analyze_token_indices()")
        print(f"  Used indices: {used_indices} ({used_indices/total_codebook_size*100:.2f}%)")
        print(f"  Unused indices: {unused_indices} ({unused_indices/total_codebook_size*100:.2f}%)")
        
        # 如果观察到的最大index接近codebook大小，说明可能使用了大部分codebook
        if max_idx_observed >= total_codebook_size - 1:
            print(f"  Note: Observed indices cover the full codebook range")
        elif max_idx_observed < total_codebook_size * 0.5:
            print(f"  ⚠️  WARNING: Only using lower {max_idx_observed/total_codebook_size*100:.1f}% of codebook range")
        
        if used_indices < total_codebook_size * 0.1:
            print(f"  ⚠️  WARNING: Less than 10% of codebook is being used!")
    
    # 显示最常用的 indices
    top_k_global = min(20, len(all_counts))
    top_global = torch.topk(all_counts, top_k_global)
    print(f"\n  Top {top_k_global} most used indices globally:")
    for j, (idx, count) in enumerate(zip(top_global.indices, top_global.values)):
        print(f"    {j+1}. Index {idx.item()}: {count.item()} times ({count.item()/len(all_indices_flat)*100:.2f}%)")
    
    print("\n" + "="*60)


def inference_full_dataset(
    eeg_path,
    coord_path,
    checkpoint_path,
    target_time=480,
    num_windows=2,
    device='cuda',
    analyze_indices=True
):
    """
    对 full_dataset 中的 1400 个样本全部做 inference
    返回聚合后的结果，并保存到本地文件
    """
    # ----- 1. 加载 full_dataset & coord -----
    full_dataset = np.load(eeg_path)          # (7, 40, 5, 62, 400)
    full_dataset = full_dataset.reshape(-1, 62, 400)  # (1400, 62, 400)
    num_samples = full_dataset.shape[0]

    coord_data = np.load(coord_path)          # (62, 3)
    coord = torch.from_numpy(coord_data.astype(np.float32)).to(device)
    coord = coord.unsqueeze(0)                # (1, 62, 3)

    # ----- 2. 加载模型 -----
    brain_tokenizer = load_model(checkpoint_path, device=device)

    # 这些变量在第一次 forward 后，根据实际 shape 动态初始化
    all_quant = None
    all_quant_per_window = None
    all_pool = None
    emb_loss_list = []
    info_list = []

    # ----- 3. 逐样本 inference -----
    for idx in tqdm(range(num_samples)):
        eeg_np = full_dataset[idx]  # (62, 400)
        eeg_tensor = torch.from_numpy(eeg_np.astype(np.float32)).unsqueeze(0).to(device)  # (1, 62, 400)

        eeg_processed, coord_processed = preprocess_data(
            eeg_tensor, coord,
            target_time=target_time,
            num_windows=num_windows
        )

        result = inference(brain_tokenizer, eeg_processed, coord_processed)

        # 把结果先搬到 CPU 再存，避免 GPU 占用太大
        quant = result['quant'].detach().cpu()
        quant_per_window = result['quant_per_window'].detach().cpu()
        pool = result['pool'].detach().cpu()
        emb_loss = torch.as_tensor(result['emb_loss']).detach().cpu()

        # 第一次 forward 时，根据 shape 初始化大 tensor
        if all_quant is None:
            all_quant = torch.empty(
                (num_samples,) + quant.shape,
                dtype=quant.dtype
            )
            all_quant_per_window = torch.empty(
                (num_samples,) + quant_per_window.shape,
                dtype=quant_per_window.dtype
            )
            all_pool = torch.empty(
                (num_samples,) + pool.shape,
                dtype=pool.dtype
            )

        all_quant[idx] = quant
        all_quant_per_window[idx] = quant_per_window
        all_pool[idx] = pool
        emb_loss_list.append(emb_loss)
        info_list.append(result['info'])

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{num_samples} samples")

    all_emb_loss = torch.stack(emb_loss_list)  # (num_samples, ...)
    
    # ----- 3.5. 分析 token indices -----
    if analyze_indices:
        analyze_token_indices(info_list, num_samples_to_analyze=20, brain_tokenizer=brain_tokenizer)

    # ----- 4. 聚合所有结果 -----
    all_results = {
        # (1400, 124, 4, 30) 形状类似，具体视模型而定
        'quant': all_quant,
        # (1400, num_windows, feature_dim) -> 每个 sample 的 window feature
        'quant_per_window': all_quant_per_window,
        # (1400, ...) 视 model pool 输出形状而定
        'pool': all_pool,
        # (1400,) 或类似
        'emb_loss': all_emb_loss,
        # list 长度 1400，每个是 encode 输出的 info
        'info': info_list,
    }

    # ----- 5. 保存到本地 -----
    torch.save(all_results, "brain_tokenizer_results.pt")
    print("Saved all results to brain_tokenizer_results.pt")

    return all_results


def main():
    # Configuration
    EEG_FOLDER = '/home/liujingqi/dataset/brain/data/Preprocessing/Segmented_Rawf_200Hz_2s'
    OUTPUT_FOLDER = '/home/liujingqi/code/eeg-test-wn/eeg_tokenizer_results_wn'
    COORD_PATH = '/share/project/BrainToken/coord/seed_dv_custom.npy'
    CHECKPOINT_PATH = '/share/project/BrainToken/all_saves/brainclip_stage1/last.ckpt'
    
    TARGET_TIME = 480
    NUM_WINDOWS = 2  # Number of temporal windows
    
    # Create output folder if not exists
    import os
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get all .npy files in the folder
    eeg_files = sorted([f for f in os.listdir(EEG_FOLDER) if f.endswith('.npy')])
    
    if len(eeg_files) == 0:
        print(f"No .npy files found in {EEG_FOLDER}")
        return None
    
    print(f"Found {len(eeg_files)} EEG files to process")
    print("="*60)
    
    # Process each file
    all_file_results = {}
    for i, eeg_file in enumerate(eeg_files):
        eeg_path = os.path.join(EEG_FOLDER, eeg_file)
        output_name = eeg_file.replace('.npy', '_results.pt')
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        
        print(f"\n[{i+1}/{len(eeg_files)}] Processing: {eeg_file}")
        print("-"*60)
        
        # 跑整套样本
        results = inference_full_dataset(
            eeg_path,
            COORD_PATH,
            CHECKPOINT_PATH,
            target_time=TARGET_TIME,
            num_windows=NUM_WINDOWS,
            device='cuda'
        )
        
        # Save individual file results
        torch.save(results, output_path)
        print(f"Saved results to: {output_path}")
        
        print(f"  quant shape:           {results['quant'].shape}")
        print(f"  quant_per_window:      {results['quant_per_window'].shape}")
        print(f"  pool shape:            {results['pool'].shape}")
        print(f"  emb_loss shape:        {results['emb_loss'].shape}")
        
        all_file_results[eeg_file] = results
    
    print("\n" + "="*60)
    print(f"All {len(eeg_files)} files processed successfully!")
    print(f"Results saved to: {OUTPUT_FOLDER}")
    
    return all_file_results


def test_random_inputs(
    coord_path,
    checkpoint_path,
    num_samples=100,
    target_time=480,
    num_windows=2,
    device='cuda'
):
    """
    使用随机输入（0-1之间的随机数）测试tokenizer
    用于验证是输入数据问题还是模型问题
    """
    print("="*60)
    print("TESTING WITH RANDOM INPUTS (0-1)")
    print("="*60)
    print(f"Generating {num_samples} random samples...")
    
    # ----- 1. 加载 coord -----
    coord_data = np.load(coord_path)          # (62, 3)
    coord = torch.from_numpy(coord_data.astype(np.float32)).to(device)
    coord = coord.unsqueeze(0)                # (1, 62, 3)
    
    # ----- 2. 加载模型 -----
    brain_tokenizer = load_model(checkpoint_path, device=device)
    
    # ----- 3. 生成随机输入并推理 -----
    info_list = []
    emb_loss_list = []
    
    for idx in tqdm(range(num_samples), desc="Processing random samples"):
        # 生成随机输入: (1, 62, 400)，值在 [0, 1] 之间
        eeg_random = np.random.rand(1, 62, 400).astype(np.float32)
        eeg_tensor = torch.from_numpy(eeg_random).to(device)  # (1, 62, 400)
        
        # 预处理
        eeg_processed, coord_processed = preprocess_data(
            eeg_tensor, coord,
            target_time=target_time,
            num_windows=num_windows
        )
        
        # 推理
        result = inference(brain_tokenizer, eeg_processed, coord_processed)
        
        # 收集结果
        emb_loss = torch.as_tensor(result['emb_loss']).detach().cpu()
        emb_loss_list.append(emb_loss)
        info_list.append(result['info'])
    
    all_emb_loss = torch.stack(emb_loss_list)
    
    # ----- 4. 分析token分布 -----
    print("\n" + "="*60)
    print("RANDOM INPUT TOKEN DISTRIBUTION ANALYSIS")
    print("="*60)
    analyze_token_indices(info_list, num_samples_to_analyze=min(20, num_samples), brain_tokenizer=brain_tokenizer)
    
    # ----- 5. 额外统计：与真实数据对比 -----
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # 提取所有indices
    all_indices = []
    for info in info_list:
        if info is not None and len(info) > 2:
            indices = info[2]
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu()
            all_indices.append(indices)
    
    if len(all_indices) > 0:
        all_indices_flat = torch.cat([idx.flatten() for idx in all_indices])
        all_unique = torch.unique(all_indices_flat)
        all_indices_long = all_indices_flat.long()
        max_idx_observed = all_indices_long.max().item()
        min_idx_observed = all_indices_long.min().item()
        
        # 从模型中获取真实的codebook大小
        actual_codebook_size = None
        codebook_source = None
        try:
            quantizer = None
            if hasattr(brain_tokenizer, 'quantize'):
                quantizer = brain_tokenizer.quantize
            elif hasattr(brain_tokenizer, 'module') and hasattr(brain_tokenizer.module, 'quantize'):
                quantizer = brain_tokenizer.module.quantize
            
            if quantizer is not None:
                if hasattr(quantizer, 're_embed'):
                    actual_codebook_size = quantizer.re_embed
                    codebook_source = "model.quantize.re_embed (remapped codebook size)"
                elif hasattr(quantizer, 'n_e'):
                    actual_codebook_size = quantizer.n_e
                    codebook_source = "model.quantize.n_e (original codebook size)"
                elif hasattr(quantizer, 'embedding') and hasattr(quantizer.embedding, 'num_embeddings'):
                    actual_codebook_size = quantizer.embedding.num_embeddings
                    codebook_source = "model.quantize.embedding.num_embeddings"
        except Exception as e:
            print(f"  ⚠️  Warning: Could not get codebook size from model: {e}")
        
        # 使用真实的codebook大小或推断值
        if actual_codebook_size is not None:
            total_codebook_size = actual_codebook_size
            all_counts = torch.bincount(all_indices_long, minlength=total_codebook_size)
        else:
            total_codebook_size = max_idx_observed + 1
            all_counts = torch.bincount(all_indices_long, minlength=total_codebook_size)
        
        used_indices = (all_counts > 0).sum().item()
        total_tokens = len(all_indices_flat)
        
        print(f"Random Input Results:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Unique indices used: {len(all_unique)}")
        print(f"  Index range: [{min_idx_observed}, {max_idx_observed}]")
        print(f"  Codebook size: {total_codebook_size}")
        if actual_codebook_size is not None and codebook_source is not None:
            print(f"    ✓ Retrieved from model: {codebook_source}")
        else:
            print(f"    ⚠️  (Inferred from max observed index: {max_idx_observed} + 1)")
        print(f"  Codebook usage: {used_indices}/{total_codebook_size} ({used_indices/total_codebook_size*100:.2f}%)")
        
        # 计算熵
        probs = all_counts.float() / all_counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log2(probs)).sum().item()
        max_entropy = np.log2(len(all_counts[all_counts > 0]))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        print(f"  Normalized entropy: {normalized_entropy:.4f}")
        
        # 显示top indices
        top_k = min(10, len(all_counts[all_counts > 0]))
        top_indices = torch.topk(all_counts, top_k)
        print(f"\n  Top {top_k} most used indices:")
        for j, (idx, count) in enumerate(zip(top_indices.indices, top_indices.values)):
            print(f"    {j+1}. Index {idx.item()}: {count.item()} times ({count.item()/total_tokens*100:.2f}%)")
        
        # 诊断
        print(f"\n" + "-"*60)
        print("DIAGNOSIS:")
        print("-"*60)
        if len(all_unique) < 10:
            print("❌ CRITICAL: Random inputs also produce very few unique tokens!")
            print("   This suggests the problem is with the MODEL, not the input data.")
            print("   Possible causes:")
            print("   - Encoder is not working properly")
            print("   - Codebook quantization is collapsing")
            print("   - Model weights may not be properly loaded")
        elif normalized_entropy < 0.3:
            print("⚠️  WARNING: Random inputs produce low diversity tokens")
            print("   The model may not be properly trained or codebook is too small")
        else:
            print("✅ Random inputs produce diverse tokens")
            print("   This suggests the problem is with the INPUT DATA, not the model.")
            print("   The real EEG data may be too similar or preprocessed incorrectly.")
    
    return {
        'info_list': info_list,
        'emb_loss': all_emb_loss,
        'num_samples': num_samples
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-random', action='store_true', 
                       help='Test with random inputs instead of real data')
    parser.add_argument('--num-random-samples', type=int, default=100,
                       help='Number of random samples to test')
    args = parser.parse_args()
    
    if args.test_random:
        COORD_PATH = '/share/project/BrainToken/coord/seed_dv_custom.npy'
        CHECKPOINT_PATH = '/share/project/BrainToken/all_saves/brainclip_stage1/last.ckpt'
        
        test_random_inputs(
            coord_path=COORD_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            num_samples=args.num_random_samples,
            target_time=480,
            num_windows=2,
            device='cuda'
        )
    else:
        results = main()


'''
Saved all results to brain_tokenizer_results.pt
All quant shape:          torch.Size([1400, 124, 4, 30])
All quant_per_window:     torch.Size([1400, 2, 7440])
All pool shape:           torch.Size([1400, 2, 2])
All emb_loss shape:       torch.Size([1400])

'''