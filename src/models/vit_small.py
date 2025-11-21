import timm

def build_vit(model_name="vit_small_patch16_384", num_classes=4):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model
