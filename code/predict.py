import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import sys
from utils import extract_mfcc_with_cache, extract_wav2vec_features, MSTRModel, EmotionAudioDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_emotion(file_path, model, label_map, device, processor=None, wav2vec_model=None, max_len=100, confidence_threshold=50, cache_dir="./mfcc_cache"):
    model.eval()
    file_path = os.path.abspath(file_path)
    use_wav2vec = processor is not None and wav2vec_model is not None
    if use_wav2vec:
        x = extract_wav2vec_features(file_path, processor, wav2vec_model, device, max_len=max_len, cache_dir=cache_dir)
    else:
        x = extract_mfcc_with_cache(file_path, max_len=max_len, cache_dir=cache_dir)
    if x.sum() == 0:
        raise ValueError(f"Failed to extract audio from {file_path}. Ensure file has valid audio track")
    
    logger.info(f"Test features shape: {x.shape}, mean: {x.mean():.4f}, std: {x.std():.4f}")
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        logger.info(f"Logits: {out.cpu().numpy()}")
        logger.info(f"Probabilities: {probs.cpu().numpy()}")
        max_prob, pred = torch.max(probs, 1)
    
    inv_label_map = {v: k for k, v in label_map.items()}
    confidence_score = max_prob.item() * 100

    labels = [inv_label_map[i] for i in range(len(inv_label_map))]
    prob_values = [prob.item() * 100 for prob in probs[0]]
    sorted_data = sorted(zip(labels, prob_values), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_probs = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_labels, sorted_probs)
    plt.xlabel('Emotions')
    plt.ylabel('Probability (%)')
    plt.title(f'Emotion Prediction Probabilities for {os.path.basename(file_path)}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('emotion_probabilities.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Biểu đồ đã được lưu thành emotion_probabilities.png")
    
    sorted_probs = sorted(zip(labels, prob_values), key=lambda x: x[1], reverse=True)
    logger.info("Sorted emotion probabilities:")
    for emotion, prob in sorted_probs:
        if prob > 0.01:
            logger.info(f"{emotion}: {prob:.2f}%")
    
    if confidence_score < confidence_threshold:
        logger.warning(f"Low confidence ({confidence_score:.2f}%). Prediction may be unreliable.")
        return "unknown", confidence_score
    
    predicted_emotion = inv_label_map[pred.item()]
    return predicted_emotion, confidence_score

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_wav2vec = torch.cuda.is_available()
    processor = None
    wav2vec_model = None
    if use_wav2vec:
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        except ImportError:
            logging.warning("Thư viện transformers không được cài đặt. Chuyển về MFCC.")
            use_wav2vec = False
    
    dataset = EmotionAudioDataset('./merged_dataset', max_len=100, use_wav2vec=use_wav2vec, processor=processor, wav2vec_model=wav2vec_model)
    if len(dataset) == 0:
        logger.error("No data found. Exiting.")
        exit()
    
    model = MSTRModel(
        input_dim=768 if use_wav2vec else 40,
        num_classes=len(dataset.label_map),
        num_scales=3,
        window_size=4,
        num_heads=4
    ).to(device)
    
    checkpoint_path = './models/best_model_wav2vec.pth' if use_wav2vec else './models/best_model_mfcc.pth'
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info(f"Loaded pre-trained model from {checkpoint_path}")
        except RuntimeError as e:
            logger.error(f"Error loading model: {e}. Exiting.")
            exit()
    else:
        logger.error(f"No checkpoint found at {checkpoint_path}. Please train the model first.")
        exit()

    file_path = sys.argv[1] if len(sys.argv) > 1 else "./test_audio.wav"
    if os.path.exists(file_path):
        try:
            predicted_emotion, confidence = predict_emotion(
                file_path, model, dataset.label_map, device, processor=processor, wav2vec_model=wav2vec_model,
                cache_dir="./wav2vec_cache" if use_wav2vec else "./mfcc_cache"
            )
            logger.info(f"Predicted emotion for {file_path}: {predicted_emotion} with confidence {confidence:.2f}%")
        except ValueError as e:
            logger.error(str(e))
    else:
        logger.error(f"Test file {file_path} not found. Please provide a valid audio file (.wav, .mp3, .ogg, .flac)")