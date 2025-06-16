import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
import pandas as pd
import plotly.express as px
import os

# Streamlit Config
st.set_page_config(
    page_title="Knowledge Distillation - Model Comparison - Weather Phenomena Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Image Transformation function to prepare images for model input
def transform(img, img_size=(224, 224)):
    """
    Resizes, converts to numpy array, transposes to C,H,W format,
    converts to float tensor, normalizes, and adds batch dimension.
    """
    img = img.resize(img_size) # Resize the image to the target size
    img = np.array(img)[..., :3] # Convert to numpy array and ensure 3 channels (RGB)
    img = torch.tensor(img).permute(2, 0, 1).float() # Transpose to (C, H, W) and convert to float tensor
    normalized_img = img / 255.0 # Normalize pixel values to [0, 1]
    return normalized_img.unsqueeze(0) # Add a batch dimension (B, C, H, W)

# Dictionary mapping class indices to weather phenomena names
classes = {0: 'dew',
           1: 'fogsmog',
           2: 'frost',
           3: 'glaze',
           4: 'hail',
           5: 'lightning',
           6: 'rain',
           7: 'rainbow',
           8: 'rime',
           9: 'sandstorm',
           10: 'snow'}

# ResidualBlock class for ResNet architecture
class ResidualBlock(nn.Module):
    """
    Implements a basic residual block with two convolutional layers,
    batch normalization, and ReLU activation. Includes a downsample
    path for changing dimensions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False # Bias often omitted with BatchNorm
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        # Downsample path if input/output channels or stride differ
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU() # Activation function

    def forward(self, x):
        shortcut = x.clone() # Store input for shortcut connection
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x += self.downsample(shortcut) # Add shortcut to main path
        x = self.relu(x) # Final activation

        return x

# ResNet model class
class ResNet(nn.Module):
    """
    Implements a simplified ResNet architecture.
    """
    def __init__(self, residual_block, n_blocks_lst, n_classes):
        super(ResNet, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create layers of residual blocks
        self.conv2 = self.create_layer(
            residual_block, 64, 64, n_blocks_lst[0], 1)
        self.conv3 = self.create_layer(
            residual_block, 64, 128, n_blocks_lst[1], 2)
        self.conv4 = self.create_layer(
            residual_block, 128, 256, n_blocks_lst[2], 2)
        self.conv5 = self.create_layer(
            residual_block, 256, 512, n_blocks_lst[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1) # Global average pooling
        self.flatten = nn.Flatten() # Flatten for fully connected layer
        self.fc1 = nn.Linear(512, n_classes) # Fully connected output layer

    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        """
        Helper function to create a sequential layer of residual blocks.
        """
        blocks = []
        # First block in a layer might have different stride/channels
        first_block = residual_block(in_channels, out_channels, stride)
        blocks.append(first_block)

        # Subsequent blocks in the layer maintain dimensions
        for idx in range(1, n_blocks):
            block = residual_block(out_channels, out_channels, stride=1)
            blocks.append(block)

        block_sequential = nn.Sequential(*blocks) # Combine blocks into a sequential module

        return block_sequential

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.maxpool(x)
        x = self.relu(x) # ReLU after initial maxpool, common variant
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x

# --- Model Loading ---
n_classes = len(classes)
# Determine the device to run the models on (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize and load the KD Student model
model_1 = ResNet(
    residual_block=ResidualBlock,
    n_blocks_lst=[2, 2, 2, 2], # Block configuration for KD Student
    n_classes=n_classes
).to(device)
# CORRECTED PATH: "./kdsamedata_wt.pt" instead of "./model/kdsamedata_wt.pt"
model_1.load_state_dict(torch.load(
    "./kdsamedata_wt.pt", map_location=device)) # Load pre-trained weights
model_1.eval() # Set model to evaluation mode (disables dropout, batchnorm updates)

# Initialize and load the Teacher model
model_2 = ResNet(
    residual_block=ResidualBlock,
    n_blocks_lst=[3, 4, 6, 3], # Block configuration for Teacher model
    n_classes=n_classes
).to(device)
# CORRECTED PATH: "./teacher_wt.pt" instead of "./model/teacher_wt.pt"
model_2.load_state_dict(torch.load(
    "./teacher_wt.pt", map_location=device)) # Load pre-trained weights
model_2.eval() # Set model to evaluation mode

# --- Streamlit Application UI ---
st.title("Weather Phenomena Prediction - Model Comparison")

# Predefined image sets for easy testing
image_sets = {
    "Set 5 images": "./static/set5",
    "Set 10 images": "./static/set10",
    "Set 15 images": "./static/set15"
}
st.markdown("<hr style='border: 1px solid #ccc; margin: 20px 0;'>", unsafe_allow_html=True)
# Option to use predefined image set or upload custom images
st.subheader("Select Predefined Image Set or Upload Your Own Images")
use_predefined_set = st.radio("Choose an option:", ["Predefined Set", "Upload Images"])

images = [] # Initialize an empty list to hold images

if use_predefined_set == "Predefined Set":
    selected_set = st.selectbox("Choose a predefined set:", list(image_sets.keys()))
    image_folder = image_sets[selected_set]
    # Check if the directory exists before listing files
    if os.path.exists(image_folder):
        # List image files in the selected folder
        image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
                       if img.lower().endswith(('jpg', 'jpeg', 'png'))]
        images = [Image.open(img_path) for img_path in image_paths]
    else:
        st.warning(f"Image folder '{image_folder}' not found. Please ensure it exists in your project.")
        images = [] # Ensure images list is empty if folder not found
else: # User chooses to upload images
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    images = [Image.open(img) for img in uploaded_files] if uploaded_files else []

st.markdown("<hr style='border: 1px solid #ccc; margin: 20px 0;'>", unsafe_allow_html=True)

if images:
    st.write(f"**Number of images selected:** {len(images)}")

    # Display images in a grid for preview
    st.subheader("Image Set Preview:")
    cols = st.columns(5) # Create 5 columns for image display
    for idx, img in enumerate(images):
        with cols[idx % 5]: # Distribute images across columns
            st.image(img, use_container_width=True, caption=f"Image {idx + 1}")

    results = [] # List to store prediction results
    total_time_1, total_time_2 = 0, 0 # Initialize total prediction times

    with st.spinner("Running Predictions... Please wait."):
        for idx, img in enumerate(images):
            # Transform image for model input
            input_tensor = transform(img).to(device)

            # Predictions for KD Student model
            start_time_1 = time.time()
            with torch.no_grad(): # Disable gradient calculation for inference
                output_1 = model_1(input_tensor)
                _, predicted_class_1 = torch.max(output_1, 1) # Get the predicted class index
            end_time_1 = time.time()
            predicted_label_1 = classes[predicted_class_1.item()] # Map index to label
            time_1 = end_time_1 - start_time_1 # Calculate inference time
            total_time_1 += time_1 # Accumulate total time

            # Predictions for Teacher model
            start_time_2 = time.time()
            with torch.no_grad():
                output_2 = model_2(input_tensor)
                _, predicted_class_2 = torch.max(output_2, 1)
            end_time_2 = time.time()
            predicted_label_2 = classes[predicted_class_2.item()]
            time_2 = end_time_2 - start_time_2
            total_time_2 += time_2

            # Store results for both models for the current image
            results.append({
                "Image": f"Image {idx + 1}",
                "Model": "KD Student",
                "Prediction": predicted_label_1,
                "Time Taken (s)": time_1
            })
            results.append({
                "Image": f"Image {idx + 1}",
                "Model": "Teacher",
                "Prediction": predicted_label_2,
                "Time Taken (s)": time_2
            })

    # Convert results to a Pandas DataFrame for easy display and plotting
    results_df = pd.DataFrame(results)

    st.markdown("<hr style='border: 1px solid #ccc; margin: 20px 0;'>", unsafe_allow_html=True)
    # Display prediction results in separate tables for KD Student and Teacher
    st.subheader("Prediction Results:")
    results_kd = results_df[results_df["Model"] == "KD Student"].reset_index(drop=True)
    results_teacher = results_df[results_df["Model"] == "Teacher"].reset_index(drop=True)

    # Use columns to display tables side-by-side
    col_kd, col_teacher = st.columns([0.5, 0.5])
    with col_kd:
        st.write("**KD Student Results**")
        st.dataframe(results_kd)
    with col_teacher:
        st.write("**Teacher Results**")
        st.dataframe(results_teacher)

    st.markdown("<hr style='border: 1px solid #ccc; margin: 20px 0;'>", unsafe_allow_html=True)
    # Display comparison charts
    st.subheader("Comparison Charts:")
    chart_col1, chart_col2 = st.columns([0.3, 0.7])

    with chart_col1:
        # Total time comparison bar chart
        total_times_df = pd.DataFrame({
            "Model": ["KD Student", "Teacher"],
            "Total Time (s)": [total_time_1, total_time_2]
        })
        total_fig = px.bar(
            total_times_df,
            x="Model",
            y="Total Time (s)",
            text="Total Time (s)", # Display text on bars
            title="Total Time Comparison",
            labels={"Total Time (s)": "Time (seconds)"}
        )
        total_fig.update_traces(texttemplate='%{text:.4f}', textposition='outside') # Format text
        st.plotly_chart(total_fig, use_container_width=True)

    with chart_col2:
        # Per-image time comparison bar chart
        fig = px.bar(
            results_df,
            x="Image",
            y="Time Taken (s)",
            color="Model", # Differentiate bars by model
            barmode="group", # Group bars for each image
            title="Time Comparison for Each Image",
            labels={"Time Taken (s)": "Time (seconds)"}
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class="footer">
        Made by <a href="https://github.com/melanieyes" target="_blank">Melanie Github</a>
    </div>
    """,
    unsafe_allow_html=True
)
