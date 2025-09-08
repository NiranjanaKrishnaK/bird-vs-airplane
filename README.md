# Bird vs Airplane Classifier (PyTorch)

This project trains a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset as either **birds** or **airplanes**.

## Dataset

- **Source**: CIFAR-10 (10 classes, 60,000 images)  
- **Filtered**: Only 2 classes → airplane and bird 
- **Size**: 6,000 images per class (balanced)  
- **Format**: RGB, 32×32 resolution  

Sample images:  
![Exploration](results/exploration.png)

## Model

- 2 convolutional layers (with ReLU activations + MaxPooling)  
- Flatten layer  
- Fully connected layers  
- Output: 2 classes (binary classification)  

## Project Structure

```
├── models.py           # CNN model
├── utils.py            # Dataset loader + plotting functions
├── train.py            # Training + validation loop
├── exploration.py      # Dataset visualization
├── requirements.txt    # Dependencies
├── results/ (exploration.png file and metrices.png file)            # Metrics & plots
└── README.md           # Project description
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bird-vs-airplane.git
   cd bird-vs-airplane
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run training:
   ```bash
   python train.py
   ```

4. Explore dataset:
   ```bash
   python exploration.py

   ```
