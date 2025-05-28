import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import sys
from utils import extract_mfcc_with_cache, MSTRModel, EmotionAudioDataset

def predict_emotion(file_path, model, label_map, device, max_len=100, confidence_threshold=80):
    model.eval()
    x = extract_mfcc_with_cache(file_path, max_len=max_len)
    if x.sum() == 0:
        raise ValueError(f"failed to extract audio from {file_path}. ensure file has valid audio track")
    
    print(f"test mfcc shape: {x.shape}, mean: {x.mean():.4f}, std: {x.std():.4f}")
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        print(f"logits: {out.cpu().numpy()}")
        print(f"probabilities: {probs.cpu().numpy()}")
        max_prob, pred = torch.max(probs, 1)
    
    inv_label_map = {v: k for k, v in label_map.items()}
    confidence_score = max_prob.item() * 100

    #label
    labels = [inv_label_map[i] for i in range(len(inv_label_map))]
    prob_values = [prob.item() * 100 for prob in probs[0]]
    # sort dữ liệu để biểu đồ dễ nhìn hơn (từ cao đến thấp)
    sorted_data = sorted(zip(labels, prob_values), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_probs = zip(*sorted_data)

    # tạo biểu đồ cột
    plt.figure(figsize=(10, 6))  # kích thước biểu đồ
    plt.bar(sorted_labels, sorted_probs, color=['#ff6384' if l == 'YAF_disgust' else '#36a2eb' for l in sorted_labels])
    plt.xlabel('Emotions')
    plt.ylabel('Probability (%)')
    plt.title('Emotion Prediction Probabilities for test_audio.wav')
    plt.xticks(rotation=45, ha='right')  # xoay nhãn cho dễ đọc
    plt.tight_layout()  # tránh nhãn bị cắt

    # lưu thành file hình ảnh
    plt.savefig('emotion_probabilities.png', dpi=300, bbox_inches='tight')
    plt.close()  # đóng figure để tiết kiệm bộ nhớ

    print("biểu đồ đã được lưu thành emotion_probabilities.png")
    
    # in tất cả xác suất cho từng nhãn
    sorted_probs = sorted(zip(labels, prob_values), key=lambda x: x[1], reverse=True)
    print("sorted emotion probabilities:")
    for emotion, prob in sorted_probs:
        if prob > 0.01:  # chỉ in những xác suất > 0.01%
            print(f"{emotion}: {prob:.2f}%")
    
    if confidence_score < confidence_threshold:
        print(f"warning: low confidence ({confidence_score:.2f}%). prediction may be unreliable.")
        return "unknown", confidence_score
    
    predicted_emotion = inv_label_map[pred.item()]
    return predicted_emotion, confidence_score

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = EmotionAudioDataset('./Tess', max_len=100)
    if len(dataset) == 0:
        print("no data found. exiting.")
        exit()
    
    model = MSTRModel(
        input_dim=40,
        num_classes=len(dataset.label_map),
        num_scales=3,
        window_size=4,
        num_heads=4
    ).to(device)
    
    checkpoint_path = './models/best_model.pth'
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"loaded pre-trained model from {checkpoint_path}")
        except RuntimeError as e:
            print(f"error loading model: {e}. exiting.")
            exit()
    else:
        print(f"no checkpoint found at {checkpoint_path}. please train the model first.")
        exit()

    file_path = sys.argv[1] if len(sys.argv) > 1 else "./test_audio.wav"
    if os.path.exists(file_path):
        try:
            predicted_emotion, confidence = predict_emotion(file_path, model, dataset.label_map, device)
            print(f"predicted emotion for {file_path}: {predicted_emotion} with confidence {confidence:.2f}%")
        except ValueError as e:
            print(e)
    else:
        print(f"test file {file_path} not found. please provide a valid audio file (.wav, .mp3, .ogg, .flac)")