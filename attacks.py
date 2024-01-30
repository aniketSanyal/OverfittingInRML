def fgsm_attack(image, epsilon, data_grad):
    """
    Apply FGSM attack.
    :param image: Original image
    :param epsilon: Perturbation amount
    :param data_grad: Gradient of the loss w.r.t. input image
    :return: Perturbed image
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(model, image, label, epsilon, alpha, iters):
    """
    Apply PGD attack.
    :param model: The model to fool
    :param image: Original image
    :param label: True label of the image
    :param epsilon: Perturbation amount
    :param alpha: Step size
    :param iters: Number of iterations
    :return: Perturbed image
    """
    # Initialize perturbation
    perturbation = torch.zeros_like(image).to(device)
    perturbation.requires_grad = True

    for _ in range(iters):
        outputs = model(image + perturbation)
        loss = F.cross_entropy(outputs, label)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            # Update perturbation
            perturbation += alpha * perturbation.grad.sign()
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
    return image + perturbation
