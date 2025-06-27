import cv2
import numpy as np
import os
import glob
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ThermalPreprocessor')

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

def create_contour_visualization(original_img, box):
    """Create visualization of the detected contour"""
    contour_img = original_img.copy()
    cv2.drawContours(contour_img, [box.astype(np.int32)], -1, (0, 255, 0), 2)
    return contour_img

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
    """Main processing pipeline with refactored functions"""
    logger.info(f"Processing image: {image_path}")
    
    # 1. Read image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image at {image_path}")
        return None
    
    original = image.copy()
    
    # 2. Create binary mask
    gray, binary = create_binary_mask(image)
    
    # 3. Extract largest contour
    largest_contour = extract_largest_contour(binary)
    if largest_contour is None:
        logger.warning(f"No contours found in {os.path.basename(image_path)}!")
        return None
    
    # 4. Get ordered corner points (NO VISUALIZATION HERE)
    ordered_corners = get_and_order_corner_points(largest_contour)
    if ordered_corners is None or len(ordered_corners) != 4:
        logger.warning(f"Could not find 4 corners for {os.path.basename(image_path)}")
        return None
    
    # 5. Apply perspective transform
    warped, width, height = apply_perspective_transform(image, ordered_corners)
    
    # 6. Adjust orientation
    oriented = adjust_image_orientation(warped)
    
    # 7. Crop and resize
    processed = perform_tight_cropping(oriented, target_size)
    
    return {
        'original': original,
        'binary': binary,
        'processed': processed,
        'largest_contour': largest_contour,
        'corners': ordered_corners,  # Changed from 'box' to 'corners'
        'warped_size': (width, height)
    }

def visualize_processing_steps(results, image_path, class_name=None):
    """Visualize all processing steps in a single row"""
    if results is None:
        logger.warning("Cannot visualize invalid data")
        return
    
    # Create visualization only when needed
    contour_img = results['original'].copy()
    cv2.drawContours(contour_img, 
                    [results['corners'].astype(np.int32)], 
                    -1, (0, 255, 0), 2)

    # Create single row with 4 columns
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    img_name = os.path.basename(image_path)
    
    # 1. Original Image
    ax[0].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
    ax[0].set_title("1. Original Image")
    ax[0].axis("off")
    
    # 2. Binary Mask with Contour Points
    ax[1].imshow(results['binary'], cmap="gray")
    if results['largest_contour'] is not None:
        for point in results['largest_contour']:
            ax[1].plot(point[0][0], point[0][1], 'r.', markersize=2)
    ax[1].set_title("2. Binary Mask with Points")
    ax[1].axis("off")
    
    # 3. Contour Visualization
    ax[2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    ax[2].set_title("3. Detected Contour")
    ax[2].axis("off")
    
    # 4. Processed Image
    ax[3].imshow(cv2.cvtColor(results['processed'], cv2.COLOR_BGR2RGB))
    ax[3].set_title(f"4. Processed\n{results['processed'].shape[1]}x{results['processed'].shape[0]}")
    ax[3].axis("off")
    
    plt.suptitle(f"Class: {class_name}\nImage: {img_name}", y=1.1)
    plt.tight_layout()
    plt.show()
    
def process_dataset(input_folder, output_folder, target_size=(60, 110)):
    """Process all images in all subfolders"""
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
    
    for subfolder in subfolders:
        class_name = os.path.basename(subfolder)
        output_subfolder = os.path.join(output_folder, class_name)
        os.makedirs(output_subfolder, exist_ok=True)
        
        logger.info(f"Processing class: {class_name}")
        
        # Process images in the subfolder
        image_files = glob.glob(os.path.join(subfolder, "*.tif"))
        visualized = False
        
        for img_path in image_files:
            results = preprocess_thermal_image(img_path, target_size)
            
            if results is not None:
                output_path = os.path.join(output_subfolder, os.path.basename(img_path))
                cv2.imwrite(output_path, results['processed'])
                logger.info(f"Saved processed image for {class_name}/{os.path.basename(img_path)}")
                
                # Visualize the first successful image in each class
                if not visualized:
                    visualize_processing_steps(results, img_path, class_name)
                    visualized = True


def main():
    input_folder = r"C:\Users\tanan\Downloads\anchee_fyp2\PVF_10\PVF_10_Original"
    output_folder = r"C:\Users\tanan\Downloads\anchee_fyp2\PVF_10\PVF_10_Try"

    logger.info("Preprocessing PVF-10 dataset...")
    process_dataset(input_folder, output_folder)

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()