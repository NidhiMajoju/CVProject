# Seeing Through the Storm: Synthetic Weather Augmentation for Robust Road Scene Segmentation

## Project Description/Statement
Autonomous driving systems rely on computer vision models, such as semantic segmentation, to understand road scenes. However, many driving datasets are dominated by clear daytime images, with far fewer examples captured under adverse conditions such as night, rain, or fog. As a result, models trained primarily on daytime data often perform poorly when deployed in challenging real world environments. 
Collecting and manually annotating large scale datasets for every weather and lighting condition is costly and time consuming, especially for pixel level segmentation tasks. Therefore, there is a need for methods that improve model robustness without requiring extensive new annotations. 
In this project, we address this problem by investigating whether image to image translation techniques can be used to generate adverse weather driving images from clear daytime scenes. These synthetic images are then used to augment training data and improve segmentation performance under domain shift. 
