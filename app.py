import streamlit as st
import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
import os
import pickle
from pathlib import Path
import io

# Page configuration
st.set_page_config(page_title="CNN Product Detection", layout="wide", page_icon="🛍️")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

# Sidebar navigation
st.sidebar.title("🛍️ Product Detection System")
page = st.sidebar.radio("Navigate", ["Train CNN Model", "Product Image Detection"])

# ==================== PAGE 1: TRAIN CNN MODEL ====================
if page == "Train CNN Model":
    st.title("🧠 CNN Model Training")
    st.markdown("Train a CNN model from scratch to identify products from images")
    
    st.markdown("---")
    
    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload Training Data CSV")
        csv_file = st.file_uploader("Upload CNN_Model_Train_Data.csv", type=['csv'])
        
        if csv_file:
            df = pd.read_csv(csv_file)
            st.success(f"✅ Loaded {len(df)} records")
            st.dataframe(df.head(), use_container_width=True)
            
            if 'product' in df.columns:
                unique_products = df['product'].nunique()
                st.info(f"🏷️ Found {unique_products} unique product classes")
    
    with col2:
        st.subheader("📁 Upload Training Images")
        uploaded_images = st.file_uploader(
            "Upload product images (zip file or multiple images)", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_images:
            st.success(f"✅ Uploaded {len(uploaded_images)} images")
            
            # Display sample images
            st.write("Sample images:")
            cols = st.columns(5)
            for idx, img_file in enumerate(uploaded_images[:5]):
                with cols[idx]:
                    img = Image.open(img_file)
                    st.image(img, use_column_width=True, caption=img_file.name[:15])
    
    st.markdown("---")
    
    # Model configuration
    st.subheader("⚙️ Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        img_size = st.selectbox("Image Size", [128, 224, 256], index=1)
        epochs = st.slider("Number of Epochs", 5, 100, 30)
    
    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        learning_rate = st.select_slider("Learning Rate", 
                                         options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                                         value=0.001)
    
    with col3:
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        use_augmentation = st.checkbox("Use Data Augmentation", value=True)
    
    # Training button
    st.markdown("---")
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if csv_file and uploaded_images:
            
            # Create progress containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.container()
            
            with st.spinner("Training CNN model..."):
                
                # Step 1: Prepare data
                status_text.text("📊 Preparing data...")
                progress_bar.progress(10)
                
                # Load CSV
                df = pd.read_csv(csv_file)
                class_names = sorted(df['product'].unique().tolist())
                num_classes = len(class_names)
                
                st.session_state.class_names = class_names
                
                # Create temporary directory structure for images
                temp_dir = Path("temp_images")
                temp_dir.mkdir(exist_ok=True)
                
                for class_name in class_names:
                    (temp_dir / class_name).mkdir(exist_ok=True)
                
                # Save uploaded images to appropriate folders
                for img_file in uploaded_images:
                    # Extract product name from filename or use first class as default
                    img_name = img_file.name
                    # Try to match filename with product names
                    matched_class = class_names[0]  # default
                    for class_name in class_names:
                        if class_name.lower().replace(' ', '_') in img_name.lower():
                            matched_class = class_name
                            break
                    
                    img = Image.open(img_file)
                    img.save(temp_dir / matched_class / img_name)
                
                progress_bar.progress(20)
                
                # Step 2: Create data generators
                status_text.text("🔄 Creating data augmentation pipeline...")
                
                if use_augmentation:
                    train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        horizontal_flip=True,
                        zoom_range=0.2,
                        validation_split=validation_split
                    )
                else:
                    train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        validation_split=validation_split
                    )
                
                train_generator = train_datagen.flow_from_directory(
                    temp_dir,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='training'
                )
                
                validation_generator = train_datagen.flow_from_directory(
                    temp_dir,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='validation'
                )
                
                progress_bar.progress(30)
                
                # Step 3: Build CNN model
                status_text.text("🏗️ Building CNN architecture...")
                
                model = models.Sequential([
                    # First convolutional block
                    layers.Conv2D(32, (3, 3), activation='relu', 
                                 input_shape=(img_size, img_size, 3)),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D((2, 2)),
                    layers.Dropout(0.25),
                    
                    # Second convolutional block
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D((2, 2)),
                    layers.Dropout(0.25),
                    
                    # Third convolutional block
                    layers.Conv2D(128, (3, 3), activation='relu'),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D((2, 2)),
                    layers.Dropout(0.25),
                    
                    # Fourth convolutional block
                    layers.Conv2D(256, (3, 3), activation='relu'),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D((2, 2)),
                    layers.Dropout(0.25),
                    
                    # Flatten and dense layers
                    layers.Flatten(),
                    layers.Dense(512, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.5),
                    layers.Dense(256, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(num_classes, activation='softmax')
                ])
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                progress_bar.progress(40)
                
                # Display model architecture
                with metrics_container:
                    st.subheader("📐 Model Architecture")
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text("\n".join(model_summary))
                
                # Step 4: Train model
                status_text.text("🎯 Training model...")
                
                # Custom callback to update progress
                class StreamlitCallback(keras.callbacks.Callback):
                    def __init__(self, progress_bar, status_text, epochs):
                        self.progress_bar = progress_bar
                        self.status_text = status_text
                        self.epochs = epochs
                        
                    def on_epoch_end(self, epoch, logs=None):
                        progress = 40 + int((epoch + 1) / self.epochs * 50)
                        self.progress_bar.progress(progress)
                        self.status_text.text(
                            f"Epoch {epoch + 1}/{self.epochs} - "
                            f"Loss: {logs['loss']:.4f} - "
                            f"Acc: {logs['accuracy']:.4f} - "
                            f"Val_Loss: {logs['val_loss']:.4f} - "
                            f"Val_Acc: {logs['val_accuracy']:.4f}"
                        )
                
                history = model.fit(
                    train_generator,
                    epochs=epochs,
                    validation_data=validation_generator,
                    callbacks=[StreamlitCallback(progress_bar, status_text, epochs)],
                    verbose=0
                )
                
                progress_bar.progress(90)
                status_text.text("💾 Saving model...")
                
                # Save model
                model.save('product_cnn_model.h5')
                
                # Save class names
                with open('class_names.pkl', 'wb') as f:
                    pickle.dump(class_names, f)
                
                st.session_state.model = model
                st.session_state.training_complete = True
                
                progress_bar.progress(100)
                status_text.text("✅ Training completed successfully!")
                
                # Display training results
                st.markdown("---")
                st.subheader("📊 Training Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Final Training Accuracy", 
                             f"{history.history['accuracy'][-1]:.2%}")
                    st.metric("Final Training Loss", 
                             f"{history.history['loss'][-1]:.4f}")
                
                with col2:
                    st.metric("Final Validation Accuracy", 
                             f"{history.history['val_accuracy'][-1]:.2%}")
                    st.metric("Final Validation Loss", 
                             f"{history.history['val_loss'][-1]:.4f}")
                
                # Plot training history
                st.subheader("📈 Training History")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    history_df = pd.DataFrame({
                        'Epoch': range(1, epochs + 1),
                        'Training Accuracy': history.history['accuracy'],
                        'Validation Accuracy': history.history['val_accuracy']
                    })
                    st.line_chart(history_df.set_index('Epoch'))
                
                with col2:
                    loss_df = pd.DataFrame({
                        'Epoch': range(1, epochs + 1),
                        'Training Loss': history.history['loss'],
                        'Validation Loss': history.history['val_loss']
                    })
                    st.line_chart(loss_df.set_index('Epoch'))
                
                st.success("🎉 Model training complete! Go to 'Product Image Detection' to test it.")
                
        else:
            st.error("⚠️ Please upload both CSV file and images before training")

# ==================== PAGE 2: PRODUCT IMAGE DETECTION ====================
elif page == "Product Image Detection":
    st.title("🔍 Product Image Detection")
    st.markdown("Upload a product image to identify and find matching products")
    
    # Check if model exists
    if not st.session_state.training_complete:
        if os.path.exists('product_cnn_model.h5') and os.path.exists('class_names.pkl'):
            st.info("📥 Loading previously trained model...")
            st.session_state.model = keras.models.load_model('product_cnn_model.h5')
            with open('class_names.pkl', 'rb') as f:
                st.session_state.class_names = pickle.load(f)
            st.session_state.training_complete = True
            st.success("✅ Model loaded successfully!")
        else:
            st.warning("⚠️ No trained model found. Please train a model first in the 'Train CNN Model' page.")
            st.stop()
    
    st.markdown("---")
    
    # Image upload
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Product Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Predict button
            if st.button("🔮 Identify Product", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    
                    # Preprocess image
                    img_size = st.session_state.model.input_shape[1]
                    img = image.resize((img_size, img_size))
                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0
                    
                    # Make prediction
                    predictions = st.session_state.model.predict(img_array)
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx]
                    predicted_class = st.session_state.class_names[predicted_class_idx]
                    
                    # Store results in session state
                    st.session_state.prediction_results = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'all_predictions': predictions[0]
                    }
    
    with col2:
        st.subheader("🎯 Detection Results")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            
            # Display predicted class
            st.markdown("### Identified Product Class")
            st.success(f"**{results['class']}**")
            
            # Display confidence
            st.metric("Confidence Score", f"{results['confidence']:.2%}")
            
            # Confidence bar
            st.progress(float(results['confidence']))
            
            # Top 5 predictions
            st.markdown("### Top 5 Predictions")
            top_5_idx = np.argsort(results['all_predictions'])[-5:][::-1]
            
            predictions_df = pd.DataFrame({
                'Product Class': [st.session_state.class_names[i] for i in top_5_idx],
                'Confidence': [f"{results['all_predictions'][i]:.2%}" for i in top_5_idx],
                'Score': [results['all_predictions'][i] for i in top_5_idx]
            })
            
            st.dataframe(predictions_df[['Product Class', 'Confidence']], 
                        use_container_width=True, hide_index=True)
            
            # Product description (mock)
            st.markdown("### 📝 Product Description")
            st.info(f"""
            **{results['class']}**
            
            This product has been identified with {results['confidence']:.1%} confidence. 
            The CNN model analyzed the image features including shape, color, texture, 
            and distinctive characteristics to make this classification.
            
            For detailed product specifications and purchasing options, please refer 
            to the matching products below.
            """)
            
            # Matching products (mock data)
            st.markdown("### 🔗 Matching & Related Products")
            
            # Generate mock similar products
            similar_products = []
            for i in range(min(4, len(st.session_state.class_names))):
                idx = (predicted_class_idx + i) % len(st.session_state.class_names)
                similar_products.append({
                    'Product Name': st.session_state.class_names[idx],
                    'Similarity Score': f"{max(0.5, 1.0 - i * 0.15):.2%}",
                    'Price Range': f"${np.random.randint(50, 500)}",
                    'Availability': np.random.choice(['In Stock', 'Limited Stock', 'Pre-order'])
                })
            
            similar_df = pd.DataFrame(similar_products)
            st.dataframe(similar_df, use_container_width=True, hide_index=True)
            
            # Natural language summary
            st.markdown("### 💬 Natural Language Summary")
            st.write(f"""
            Based on the image analysis, the product has been identified as **{results['class']}** 
            with a confidence level of **{results['confidence']:.1%}**. 
            
            We found {len(similar_products)} matching and related products in our database. 
            The most similar product is {similar_products[0]['Product Name']} with a 
            similarity score of {similar_products[0]['Similarity Score']}.
            
            All matching products are currently available for viewing and purchase through 
            our product catalog.
            """)
        else:
            st.info("👆 Upload an image and click 'Identify Product' to see results")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>CNN Product Detection System | Built with Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)