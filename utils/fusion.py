def fuse_predictions(text_prob, image_prob):

    # Weighted fusion
    final_score = (0.6 * text_prob) + (0.4 * image_prob)

    if final_score < 0.3:
        category = "Low Risk"
    elif final_score < 0.7:
        category = "Medium Risk"
    else:
        category = "High Risk"

    final_pred = 1 if final_score > 0.5 else 0

    return final_pred, final_score, category