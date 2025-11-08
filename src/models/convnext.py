import timm

def build_convnext(model_name="convnext_tiny", num_classes=4):
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)
