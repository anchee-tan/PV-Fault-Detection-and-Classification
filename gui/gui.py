import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from datetime import datetime
import threading
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm
import time
import torch
import torch.nn as nn
from torchvision import models

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        # Enhanced initial block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Enhanced residual blocks with skip connections
        self.res_block1 = self._make_residual_block(32, 64)
        self.res_block2 = self._make_residual_block(64, 128)
        self.res_block3 = self._make_residual_block(128, 256)
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Enhanced classifier with more capacity
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Enhanced attention
        attention = self.attention(x)
        x = x * attention
        
        return self.classifier(x)
        
def create_customcnn_model(num_classes):
    """Create a custom CNN model from scratch"""
    print("\nCreating custom CNN model with architecture:")
    print("-------------------------------------------")
    print("Input: 3x224x224 RGB image")
    print("5 convolutional blocks with increasing filters (32, 64, 128, 256, 512)")
    print("Each block contains: Conv2d -> BatchNorm -> ReLU -> MaxPool")
    print("Classifier head with 3 fully connected layers (1024, 512, num_classes)")
    print("Dropout layers for regularization")
    
    model = CustomCNN(num_classes)
    
    # Print model summary
    print("\nModel Summary:")
    print("-------------")
    print(model)
    
    return model

# Preprocessing functions (using your latest version)
def create_binary_mask(image):
    """Convert to grayscale and create binary mask"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return gray, binary

def extract_largest_contour(binary):
    """Find and extract the largest contour"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def get_and_order_corner_points(contour):
    """
    Finds and orders 4 corner points (TL, TR, BR, BL) from a contour
    Combines get_corner_points() and order_points() logic
    """
    # Method 1: Contour approximation
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        # Method 2: Convex hull
        hull = cv2.convexHull(contour)
        if len(hull) == 4:
            corners = hull.reshape(4, 2).astype(np.float32)
        else:
            # Method 3: Minimum area rectangle
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype(np.float32)
    
    # Order points (TL, TR, BR, BL)
    # 1. Sort by x-coordinate
    x_sorted = corners[np.argsort(corners[:, 0])]
    
    # 2. Split into left and right points
    left_points = x_sorted[:2]
    right_points = x_sorted[2:]
    
    # 3. Sort left points by y-coordinate (top to bottom)
    left_points = left_points[np.argsort(left_points[:, 1])]
    tl, bl = left_points[0], left_points[1]
    
    # 4. Sort right points by y-coordinate (top to bottom)
    right_points = right_points[np.argsort(right_points[:, 1])]
    tr, br = right_points[0], right_points[1]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def apply_perspective_transform(image, box):
    """Apply perspective transformation"""
    width, height = calculate_dimensions(box)
    dst_pts = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst_pts)
    return cv2.warpPerspective(image, M, (width, height)), width, height

def adjust_image_orientation(warped):
    """Adjust image orientation based on dimensions and brightness"""
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray_warped.shape[:2]
    
    if w > h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        h, w = w, h
    
    if np.mean(gray_warped[:h//4]) > np.mean(gray_warped[3*h//4:]):
        warped = cv2.flip(warped, 0)
    
    return warped

def calculate_dimensions(box):
    """Calculate width and height from ordered points"""
    width = int(max(np.linalg.norm(box[0]-box[1]), np.linalg.norm(box[2]-box[3])))
    height = int(max(np.linalg.norm(box[1]-box[2]), np.linalg.norm(box[3]-box[0])))
    return width, height
    
def perform_tight_cropping(warped, target_size):
    """Crop tightly around the object and resize"""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = warped[y:y+h, x:x+w]
    else:
        cropped = warped
    
    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

def preprocess_thermal_image(image_path, target_size=(60, 110)):
    """Main processing pipeline"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    original = image.copy()
    gray, binary = create_binary_mask(image)
    largest_contour = extract_largest_contour(binary)
    
    if largest_contour is None:
        return None
    
    ordered_corners = get_and_order_corner_points(largest_contour)
    if ordered_corners is None or len(ordered_corners) != 4:
        return None
    
    warped, width, height = apply_perspective_transform(image, ordered_corners)
    oriented = adjust_image_orientation(warped)
    processed = perform_tight_cropping(oriented, target_size)
    
    return {
        'original': original,
        'processed': processed,
        'corners': ordered_corners,
        'warped_size': (width, height)
    }

class EnhancedThermalImageClassifierApp:
    def __init__(self, root, model, class_names, transform):
        self.root = root
        self.model = model
        self.class_names = class_names
        self.transform = transform
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Configure window
        self.root.title("🔬 Thermal Image Classifier - AI Powered Solar Panel Diagnosis")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Configure styles
        self.setup_styles()
        
        # Create main layout
        self.create_header()
        self.create_control_panel()
        self.create_main_content()
        self.create_status_bar()
        
        # Initialize variables
        self.image_paths = []
        self.processed_images = []
        self.current_index = 0
        self.output_dir = os.path.join(os.getcwd(), "output_results")
        os.makedirs(self.output_dir, exist_ok=True)
        self.update_output_label()
    
    def setup_styles(self):
        """Configure modern styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#28A745',
            'warning': '#FFC107',
            'danger': '#DC3545',
            'light': '#F8F9FA',
            'dark': '#343A40',
            'background': '#FFFFFF'
        }
        
        # Configure button styles
        style.configure('Primary.TButton', 
                       background=self.colors['primary'],
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(10, 5))
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground='white',
                       font=('Segoe UI', 10))
        
        style.configure('Header.TLabel',
                       font=('Segoe UI', 16, 'bold'),
                       background='#f0f0f0',
                       foreground=self.colors['dark'])
        
        style.configure('Subheader.TLabel',
                       font=('Segoe UI', 12, 'bold'),
                       background='#f0f0f0',
                       foreground=self.colors['primary'])
    
    def create_header(self):
        """Create modern header section"""
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="🔬 Thermal Image Classifier",
                              font=('Segoe UI', 18, 'bold'),
                              bg=self.colors['primary'],
                              fg='white')
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        subtitle_label = tk.Label(header_frame,
                                 text="AI-Powered Solar Panel Fault Detection System",
                                 font=('Segoe UI', 12),
                                 bg=self.colors['primary'],
                                 fg='white')
        subtitle_label.pack(side=tk.LEFT, padx=(0, 20), pady=20)
        
        # Device info
        device_info = f"🖥️ Device: {self.device.upper()}"
        device_label = tk.Label(header_frame,
                               text=device_info,
                               font=('Segoe UI', 10),
                               bg=self.colors['primary'],
                               fg='white')
        device_label.pack(side=tk.RIGHT, padx=20, pady=20)
    
    def create_control_panel(self):
        """Create enhanced control panel"""
        control_frame = tk.Frame(self.root, bg='white', relief=tk.RAISED, bd=1)
        control_frame.pack(fill=tk.X, padx=15, pady=(10, 5))
        
        # File operations
        file_frame = tk.Frame(control_frame, bg='white')
        file_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        upload_btn = ttk.Button(file_frame, text="📁 Upload Images", 
                               style='Primary.TButton',
                               command=self.upload_images)
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        output_btn = ttk.Button(file_frame, text="📂 Output Folder",
                               command=self.select_output_folder)
        output_btn.pack(side=tk.LEFT, padx=5)
        
        # Output info
        self.output_label = tk.Label(file_frame, text="Output: Not selected",
                                    font=('Segoe UI', 9),
                                    bg='white', fg=self.colors['dark'])
        self.output_label.pack(side=tk.LEFT, padx=10)
        
        # Navigation
        nav_frame = tk.Frame(control_frame, bg='white')
        nav_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.prev_button = ttk.Button(nav_frame, text="⬅️ Previous",
                                     command=self.show_previous, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=2)
        
        self.next_button = ttk.Button(nav_frame, text="Next ➡️",
                                     command=self.show_next, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=2)
        
        # Progress
        progress_frame = tk.Frame(control_frame, bg='white')
        progress_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.progress = ttk.Progressbar(progress_frame, length=200, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=5)
        
        self.progress_label = tk.Label(progress_frame, text="Ready",
                                      font=('Segoe UI', 9),
                                      bg='white', fg=self.colors['dark'])
        self.progress_label.pack(side=tk.RIGHT, padx=5)
    
    def create_main_content(self):
        """Create main content area with side-by-side layout"""
        main_content = tk.Frame(self.root, bg='#f0f0f0')
        main_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Create paned window for resizable sections
        paned_window = ttk.PanedWindow(main_content, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left section - Original Image (30% width)
        left_frame = ttk.Frame(paned_window, width=400)
        paned_window.add(left_frame, weight=3)
        
        # Original image section
        original_section = tk.LabelFrame(left_frame, text="📷 Original Image",
                                        font=('Segoe UI', 12, 'bold'),
                                        bg='white', fg=self.colors['dark'],
                                        relief=tk.RAISED, bd=2)
        original_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.original_label = tk.Label(original_section, bg='white',
                                      text="Upload images to begin analysis",
                                      font=('Segoe UI', 12),
                                      fg=self.colors['dark'])
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Middle section - Processed Image (30% width)
        middle_frame = ttk.Frame(paned_window, width=400)
        paned_window.add(middle_frame, weight=3)
        
        # Processed image section
        processed_section = tk.LabelFrame(middle_frame, text="⚙️ Processed Image",
                                         font=('Segoe UI', 12, 'bold'),
                                         bg='white', fg=self.colors['dark'],
                                         relief=tk.RAISED, bd=2)
        processed_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.processed_label = tk.Label(processed_section, bg='white',
                                       text="Processed images will appear here",
                                       font=('Segoe UI', 10),
                                       fg=self.colors['dark'])
        self.processed_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right section - Results (40% width)
        right_frame = ttk.Frame(paned_window, width=500)
        paned_window.add(right_frame, weight=4)
        
        # AI Prediction Section (Top)
        prediction_section = tk.LabelFrame(right_frame, text="🎯 AI Prediction Results",
                                          font=('Segoe UI', 12, 'bold'),
                                          bg='white', fg=self.colors['dark'],
                                          relief=tk.RAISED, bd=2)
        prediction_section.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        self.prediction_frame = tk.Frame(prediction_section, bg='white')
        self.prediction_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.prediction_label = tk.Label(self.prediction_frame,
                                        text="No prediction available",
                                        font=('Segoe UI', 14, 'bold'),
                                        bg='white', fg=self.colors['dark'])
        self.prediction_label.pack()
        
        self.confidence_label = tk.Label(self.prediction_frame,
                                        text="",
                                        font=('Segoe UI', 12),
                                        bg='white', fg=self.colors['secondary'])
        self.confidence_label.pack(pady=(5, 0))
        
        self.confidence_bar = ttk.Progressbar(self.prediction_frame,
                                             length=300, mode='determinate')
        self.confidence_bar.pack(pady=10)
        
        # Session Statistics Section (Middle)
        stats_section = tk.LabelFrame(right_frame, text="📊 Session Statistics",
                                     font=('Segoe UI', 12, 'bold'),
                                     bg='white', fg=self.colors['dark'],
                                     relief=tk.RAISED, bd=2)
        stats_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.stats_frame = tk.Frame(stats_section, bg='white')
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.stats_text = tk.Text(self.stats_frame, height=8, width=35,
                                 font=('Consolas', 10),
                                 bg='#f8f9fa', fg=self.colors['dark'],
                                 relief=tk.FLAT, bd=1)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Quick Actions Section (Bottom)
        actions_section = tk.LabelFrame(right_frame, text="🚀 Quick Actions",
                                       font=('Segoe UI', 12, 'bold'),
                                       bg='white', fg=self.colors['dark'],
                                       relief=tk.RAISED, bd=2)
        actions_section.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        actions_frame = tk.Frame(actions_section, bg='white')
        actions_frame.pack(fill=tk.X, padx=15, pady=15)
        
        save_btn = ttk.Button(actions_frame, text="💾 Save Results",
                             style='Success.TButton',
                             command=self.save_all_results)
        save_btn.pack(fill=tk.X, pady=2)
        
        clear_btn = ttk.Button(actions_frame, text="🗑️ Clear All",
                              command=self.clear_all)
        clear_btn.pack(fill=tk.X, pady=2)
        
        # Bottom section - Image List
        list_section = tk.LabelFrame(main_content, text="📋 Image Analysis Queue",
                                    font=('Segoe UI', 12, 'bold'),
                                    bg='white', fg=self.colors['dark'],
                                    relief=tk.RAISED, bd=2)
        list_section.pack(fill=tk.BOTH, expand=False, padx=15, pady=(5, 15))
        
        # Create frame for listbox and scrollbar
        list_container = tk.Frame(list_section, bg='white')
        list_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Listbox with scrollbar
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox = tk.Listbox(list_container, height=6,
                                       font=('Segoe UI', 10),
                                       selectmode=tk.SINGLE,
                                       yscrollcommand=scrollbar.set)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        
        self.image_listbox.bind('<<ListboxSelect>>', self.on_list_select)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Frame(self.root, bg=self.colors['dark'], height=25)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_bar,
                                    text="Ready - Upload images to begin",
                                    font=('Segoe UI', 9),
                                    bg=self.colors['dark'], fg='white')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=2)
        
        # Time display
        self.time_label = tk.Label(self.status_bar,
                                  text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                  font=('Segoe UI', 9),
                                  bg=self.colors['dark'], fg='white')
        self.time_label.pack(side=tk.RIGHT, padx=10, pady=2)
        
        # Update time every second
        self.update_time()
    
    def update_time(self):
        """Update time display"""
        self.time_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.update_time)
    
    def update_output_label(self):
        """Update the output directory label"""
        short_path = os.path.basename(self.output_dir)
        if len(self.output_dir) > 40:
            short_path = "..." + short_path[-35:]
        self.output_label.config(text=f"📂 Output: {short_path}")
    
    def select_output_folder(self):
        """Let user select output folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir = folder
            self.update_output_label()
            self.update_status("Output folder updated")
            messagebox.showinfo("✅ Success", f"Output folder set to:\n{folder}")
    
    def upload_images(self):
        """Upload and process images"""
        file_paths = filedialog.askopenfilenames(
            title="Select Thermal Images",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("TIFF Files", "*.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )
        if not file_paths:
            return
        
        self.image_paths = file_paths
        self.processed_images = []
        self.image_listbox.delete(0, tk.END)
        self.current_index = 0
        
        self.update_status(f"Processing {len(file_paths)} images...")
        
        # Process images in a thread
        threading.Thread(target=self.process_images_thread, daemon=True).start()
    
    # Update your process_images_thread method to store the inference time:
    def process_images_thread(self):
        """Thread for processing images"""
        self.progress['maximum'] = len(self.image_paths)
        self.progress['value'] = 0
        
        success_count = 0
        
        for i, path in enumerate(self.image_paths):
            try:
                # Update progress
                self.progress_label.config(text=f"Processing {i+1}/{len(self.image_paths)}")
                
                # Process image
                result = preprocess_thermal_image(path)
                if result is None:
                    continue
                
                # Convert to PIL and store
                original_pil = Image.fromarray(cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB))
                processed_pil = Image.fromarray(cv2.cvtColor(result['processed'], cv2.COLOR_BGR2RGB))
                
                # Make prediction and measure time
                prediction, probability, inference_time = self.predict_image(result['processed'])
                
                # Store results
                self.processed_images.append({
                    'original': original_pil,
                    'processed': processed_pil,
                    'prediction': prediction,
                    'probability': probability,
                    'inference_time': inference_time,  # Store inference time
                    'path': path
                })
                
                # Add to listbox with status indicator
                filename = os.path.basename(path)
                status_icon = self.get_status_icon(prediction)
                self.image_listbox.insert(tk.END, f"{status_icon} {filename} ({inference_time:.1f}ms)")
                
                success_count += 1
                
                # Update progress
                self.progress['value'] = i + 1
                self.root.update()
                
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
        
        # Update UI after processing
        self.progress_label.config(text="Complete")
        
        if self.processed_images:
            self.show_current_image()
            self.update_nav_buttons()
            self.update_statistics()
            self.update_status(f"✅ Processed {success_count} images successfully")
            
            # Show completion message
            messagebox.showinfo("🎉 Processing Complete", 
                              f"Successfully processed {success_count} out of {len(self.image_paths)} images!")
        else:
            self.update_status("❌ No images were processed successfully")
            messagebox.showwarning("⚠️ Warning", "No images could be processed. Please check your image files.")
    
    def get_status_icon(self, prediction):
        """Get status icon based on prediction"""
        if "healthy" in prediction.lower():
            return "[j]"
        elif "hot cell" in prediction.lower():
            return "[i]"
        elif "junction" in prediction.lower():
            return "[h]"
        elif "break" in prediction.lower():
            return "[g]"
        elif "dirt" in prediction.lower():
            return "[f]"
        elif "shadow" in prediction.lower():
            return "[e]"
        elif "debris cover" in prediction.lower():
            return "[d]"
        elif "string short" in prediction.lower():
            return "[c]"
        elif "short circuit panel" in prediction.lower():
            return "[b]"
        elif "substring" in prediction.lower():
            return "[a]"
        else:
            return "🔍"
    
    def measure_inference_time(self, image):
        """Measure the time taken for model inference"""
        start_time = time.time()
        
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # Apply transformations
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            top_p, top_class = torch.topk(probabilities, 1)
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return self.class_names[top_class[0]], top_p[0].item(), inference_time

    def predict_image(self, image):
        """Predict the class of an image"""
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # Apply transformations
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            top_p, top_class = torch.topk(probabilities, 1)
        
        return self.measure_inference_time(image)
    
    # Update your show_current_image method to pass inference time:
    def show_current_image(self):
        """Display the current image and results"""
        if not self.processed_images or self.current_index >= len(self.processed_images):
            return
        
        current = self.processed_images[self.current_index]
        
        # Display original and processed images
        self.display_image(self.original_label, current['original'], (500, 400))
        self.display_image(self.processed_label, current['processed'], (150, 200))
        
        # Update prediction results with enhanced display
        self.update_prediction_display(
            current['prediction'], 
            current['probability'],
            current['inference_time']  # Pass inference time
        )
        
        # Update listbox selection
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)
        
        # Update status
        filename = os.path.basename(current['path'])
        self.update_status(f"Viewing: {filename} - Inference: {current['inference_time']:.1f}ms")
    
    # Update your update_prediction_display method to show inference time:
    def update_prediction_display(self, prediction, probability, inference_time=None):
        """Update prediction display with enhanced formatting"""
        # Clean up prediction text
        clean_prediction = prediction.replace("0", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "").strip()
        
        # Update main prediction label
        self.prediction_label.config(text=clean_prediction.title())
        
        # Update confidence and time
        confidence_text = f"Confidence: {probability:.1%}"
        if inference_time is not None:
            confidence_text += f"\nInference Time: {inference_time:.1f} ms"
        self.confidence_label.config(text=confidence_text)
        
        # Update confidence bar
        self.confidence_bar['value'] = probability * 100
        
        # Color code based on confidence
        if probability > 0.8:
            color = self.colors['success']
        elif probability > 0.6:
            color = self.colors['warning']
        else:
            color = self.colors['danger']
        
        self.confidence_label.config(fg=color)
    
    def update_statistics(self):
        """Update statistics display with average inference time"""
        if not self.processed_images:
            return
        
        # Count predictions and sum inference times
        prediction_counts = {}
        total_confidence = 0
        total_inference_time = 0
        
        for img in self.processed_images:
            pred = img['prediction']
            clean_pred = pred.replace("0", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "").strip()
            
            if clean_pred in prediction_counts:
                prediction_counts[clean_pred] += 1
            else:
                prediction_counts[clean_pred] = 1
            
            total_confidence += img['probability']
            total_inference_time += img['inference_time']
        
        # Calculate averages
        avg_confidence = total_confidence / len(self.processed_images)
        avg_inference_time = total_inference_time / len(self.processed_images)
        
        # Build statistics text
        stats = []
        stats.append(f"📊 ANALYSIS SUMMARY")
        stats.append(f"{'='*30}")
        stats.append(f"Total Images: {len(self.processed_images)}")
        stats.append(f"Avg Confidence: {avg_confidence:.1%}")
        stats.append(f"Avg Inference Time: {avg_inference_time:.1f} ms")
        stats.append("")
        stats.append("🔍 FAULT DISTRIBUTION:")
        stats.append("-" * 20)
        
        # Sort by count
        sorted_counts = sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True)
        
        for pred, count in sorted_counts:
            percentage = (count / len(self.processed_images)) * 100
            stats.append(f"{pred.title()}: {count} ({percentage:.1f}%)")
        
        # Update stats display
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, "\n".join(stats))
    
    def show_next(self):
        """Show next image"""
        if self.current_index < len(self.processed_images) - 1:
            self.current_index += 1
            self.show_current_image()
            self.update_nav_buttons()
    
    def show_previous(self):
        """Show previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
            self.update_nav_buttons()
    
    def on_list_select(self, event):
        """Handle image selection from list"""
        if not self.image_listbox.curselection():
            return
        new_index = self.image_listbox.curselection()[0]
        if new_index != self.current_index:
            self.current_index = new_index
            self.show_current_image()
            self.update_nav_buttons()
    
    def update_nav_buttons(self):
        """Update navigation buttons state"""
        self.prev_button['state'] = tk.NORMAL if self.current_index > 0 else tk.DISABLED
        self.next_button['state'] = tk.NORMAL if self.current_index < len(self.processed_images) - 1 else tk.DISABLED
    
    def display_image(self, label, image, size=None):
        """Display an image in a label with improved scaling"""
        if size:
            # Maintain aspect ratio
            img_copy = image.copy()
            img_copy.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Create a new image with the exact size and center the thumbnail
            final_image = Image.new('RGB', size, (240, 240, 240))
            x = (size[0] - img_copy.width) // 2
            y = (size[1] - img_copy.height) // 2
            final_image.paste(img_copy, (x, y))
            
            img_tk = ImageTk.PhotoImage(final_image)
        else:
            img_tk = ImageTk.PhotoImage(image)
        
        label.config(image=img_tk, text="")
        label.image = img_tk
    
    def save_all_results(self):
        """Save all processed images with annotations to output directory"""
        if not self.processed_images:
            messagebox.showwarning("⚠️ Warning", "No results to save!")
            return
        
        try:
            # Create timestamped subfolder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = os.path.join(self.output_dir, f"thermal_analysis_{timestamp}")
            os.makedirs(output_folder, exist_ok=True)
            
            # Save each image
            for i, img_data in enumerate(self.processed_images):
                filename = os.path.basename(img_data['path'])
                name, ext = os.path.splitext(filename)
                
                # Save original with annotation
                original_with_text = self.annotate_image(
                    img_data['original'], 
                    img_data['prediction'], 
                    img_data['probability']
                )
                original_with_text.save(os.path.join(output_folder, f"{name}_analyzed{ext}"))
                
                # Save processed image
                img_data['processed'].save(os.path.join(output_folder, f"{name}_processed{ext}"))
            
            # Save summary report
            self.save_summary_report(output_folder)
            
            self.update_status(f"✅ Results saved to: {output_folder}")
            messagebox.showinfo("💾 Success", f"All results saved successfully!\n\nLocation: {output_folder}")
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Failed to save results:\n{str(e)}")
    
    # Update your save_summary_report method to include inference times:
    def save_summary_report(self, output_folder):
        """Save a summary report of the analysis"""
        report_path = os.path.join(output_folder, "analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("THERMAL IMAGE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Analyzed: {len(self.processed_images)}\n")
            f.write(f"AI Model: Custom CNN\n")
            f.write(f"Processing Device: {self.device.upper()}\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            total_inference_time = 0
            
            for i, img_data in enumerate(self.processed_images, 1):
                filename = os.path.basename(img_data['path'])
                f.write(f"{i:2d}. {filename}\n")
                f.write(f"    Prediction: {img_data['prediction']}\n")
                f.write(f"    Confidence: {img_data['probability']:.2%}\n")
                f.write(f"    Inference Time: {img_data['inference_time']:.1f} ms\n\n")
                total_inference_time += img_data['inference_time']
            
            # Calculate average inference time
            avg_inference_time = total_inference_time / len(self.processed_images) if self.processed_images else 0
            
            # Statistics
            f.write("\nSTATISTICS:\n")
            f.write("-" * 20 + "\n")
            
            prediction_counts = {}
            total_confidence = 0
            
            for img in self.processed_images:
                pred = img['prediction']
                if pred in prediction_counts:
                    prediction_counts[pred] += 1
                else:
                    prediction_counts[pred] = 1
                total_confidence += img['probability']
            
            f.write(f"Average Confidence: {total_confidence/len(self.processed_images):.2%}\n")
            f.write(f"Average Inference Time: {avg_inference_time:.1f} ms\n\n")
            
            f.write("Fault Distribution:\n")
            for pred, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(self.processed_images)) * 100
                f.write(f"  {pred}: {count} images ({percentage:.1f}%)\n")
    
    def annotate_image(self, image, prediction, probability):
        """Add prediction text to image with enhanced styling"""
        img = image.copy()
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Clean prediction text
        clean_prediction = prediction.replace("0", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "").strip()
        
        text = f"{clean_prediction.title()}\nConfidence: {probability:.1%}"
        
        # Get text dimensions
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)
        
        # Position at bottom
        img_width, img_height = img.size
        x = 10
        y = img_height - text_height - 20
        
        # Draw background with rounded corners effect
        padding = 8
        draw.rectangle([
            x - padding, 
            y - padding, 
            x + text_width + padding, 
            y + text_height + padding
        ], fill=(0, 0, 0, 180))
        
        # Draw text
        draw.text((x, y), text, fill="white", font=font)
        
        return img
    
    def clear_all(self):
        """Clear all images and results"""
        if messagebox.askyesno("🗑️ Clear All", "Are you sure you want to clear all images and results?"):
            self.image_paths = []
            self.processed_images = []
            self.current_index = 0
            
            # Clear UI elements
            self.image_listbox.delete(0, tk.END)
            self.original_label.config(image="", text="Upload images to begin analysis")
            self.processed_label.config(image="", text="Processed images will appear here")
            self.prediction_label.config(text="No prediction available")
            self.confidence_label.config(text="")
            self.confidence_bar['value'] = 0
            self.stats_text.delete(1.0, tk.END)
            
            # Reset buttons
            self.update_nav_buttons()
            
            # Reset progress
            self.progress['value'] = 0
            self.progress_label.config(text="Ready")
            
            self.update_status("🔄 Cleared all data - Ready for new analysis")
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

# Main function
if __name__ == "__main__":
    # Load model and class names
    model_path = r"C:\Users\tanan\Downloads\anchee_fyp2\customcnn_pv_fault_best.pth"
    class_names = [
        "01substring open circuit", 
        "02short circuit panel", 
        "03string short circuit",
        "04debris cover",
        "05shadow",
        "06bottom dirt",
        "07break",
        "08junction box heat",
        "09hot cell",
        "10healthy panel"
    ]
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # Create a fresh model with the same architecture
        model = create_customcnn_model(num_classes=10)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Extract just the model's state dict from the checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded model from checkpoint.")
        else:
            # Try direct loading (though this is likely not the case based on error)
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Start GUI
        root = tk.Tk()
        app = EnhancedThermalImageClassifierApp(root, model, class_names, transform)
        root.mainloop()
    
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        messagebox.showerror("Initialization Error", f"Failed to start application:\n{str(e)}")