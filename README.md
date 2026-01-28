# Plant Disease Recognition - Thesis Project

**Implementation and Performance Evaluation of a Lightweight Image Classification Model for Plant Disease Recognition**

## Project Overview
This thesis develops a CPU-optimized deep learning system for recognizing plant diseases from leaf images. The model is trained on 38 disease classes across 14 plant species and deployed as a web application.

## Thesis Objectives
1. Develop a lightweight CNN model optimized for CPU inference
2. Create a web-based interface for non-technical users (farmers/gardeners)
3. Evaluate performance on low-specification hardware
4. Ensure accessibility and sustainability in design

## Dataset
- **Source:** New Plant Diseases Dataset (Kaggle)
- **Plants:** 14 species (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- **Classes:** 38 total (21 diseases + 17 healthy)
- **Images:** 30,400 after balancing (400 per class)
- **Image size:** 128×128 pixels (optimized for CPU)