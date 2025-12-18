from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                               QComboBox, QStackedWidget, QDoubleSpinBox, QSpinBox)
from PySide6.QtCore import Qt, Signal

# Math & Image Processing (NumPy Only)
import numpy as np
import imageio.v2 as imageio 

# Project Modules
from modules.i_image_module import IImageModule

# PARAMETER WIDGETS 

class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError

class NoParamsWidget(BaseParamsWidget):
    """Used for Laplacian (Fixed kernel)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("No parameters for this operation.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

class GaussianParamsWidget(BaseParamsWidget):
    """Widget for Gaussian Blur."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Kernel Size (Odd number 3-15):"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(3, 15)
        self.k_spin.setSingleStep(2) 
        self.k_spin.setValue(5)
        layout.addWidget(self.k_spin)

        layout.addWidget(QLabel("Sigma (Spread):"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 10.0)
        self.sigma_spin.setValue(1.0)
        layout.addWidget(self.sigma_spin)
        
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'kernel_size': self.k_spin.value(),
            'sigma': self.sigma_spin.value()
        }

class MedianParamsWidget(BaseParamsWidget):
    """Widget for Median Filter."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Kernel Size (Odd number):"))
        
        self.k_spin = QSpinBox()
        self.k_spin.setRange(3, 15) # Median is slow, keep size small
        self.k_spin.setSingleStep(2)
        self.k_spin.setValue(3)
        layout.addWidget(self.k_spin)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'kernel_size': self.k_spin.value()}

class PosterizeParamsWidget(BaseParamsWidget):
    """Widget for Posterization."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Bit Depth Reduction (2-16):"))
        
        self.level_spin = QSpinBox()
        self.level_spin.setRange(2, 16)
        self.level_spin.setValue(4)
        layout.addWidget(self.level_spin)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'levels': self.level_spin.value()}
    
class SharpenParamsWidget(BaseParamsWidget):
    """Widget for Sharpening filter."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Sharpen Strength (0 - 5):"))
        
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(0.0, 5.0)
        self.amount_spin.setSingleStep(0.1)
        self.amount_spin.setValue(1.0) # Default strength
        layout.addWidget(self.amount_spin)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'amount': self.amount_spin.value()}

# CONTROLS WIDGET 

class HelinMenControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Helin Men's Filters</h3>"))

        layout.addWidget(QLabel("Select Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

       
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

       
        self.operations = {
            "Median Filter": MedianParamsWidget,
            "Gaussian Blur": GaussianParamsWidget,
            "Laplacian (Edge Detect)": NoParamsWidget,
            "Posterization": PosterizeParamsWidget,
            "Sharpening": SharpenParamsWidget,
        }

        
        for name, widget_class in self.operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

    def _on_apply_clicked(self):
        op_name = self.operation_selector.currentText()
        widget = self.param_widgets[op_name]
        params = widget.get_params()
        params['operation'] = op_name
        self.process_requested.emit(params)

    def _on_operation_changed(self, op_name: str):
        if op_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[op_name])

# MAIN MODULE CLASS 

class HelinMenImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Helin Men Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = HelinMenControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    #HELPER FUNCTIONS FOR MANUAL CONVOLUTION 
    
    def _apply_convolution(self, img_channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Manually applies convolution by shifting the image which is faster than nested loops. 
        
        """
        # Padding the image w/ kernel size
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        # Padding with edge values 
        padded = np.pad(img_channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        output = np.zeros_like(img_channel, dtype=float)
        
        # Iterating over the kernel (small loop), not the image (big loop), which alows using use Vectorized Operations on the whole array at once
        
        for i in range(kh):
            for j in range(kw):
                weight = kernel[i, j]
                region = padded[i : i + img_channel.shape[0], j : j + img_channel.shape[1]]
                output += region * weight
                
        return output

    def _get_gaussian_1d_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Generates a 1D Gaussian Kernel (shape 1 x Size)."""
        ax = np.linspace(-(size // 2), size // 2, size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        
        # Normalizing so sum is 1
        gauss = gauss / np.sum(gauss)
        
        # Returning as a 2D array with shape (1, size) for our convolution function
        return gauss.reshape(1, size)

    # LOAD & PROCESS IMAGE LOGIC 
    def load_image(self, file_path: str):
        try:
            # Use imageio to read the file into a numpy array
            image_data = imageio.imread(file_path)
            
            # Create simple metadata
            metadata = {'name': file_path}
            
            # Return the standard tuple expected by the app:
            # (Success?, Image Data, Metadata, Extra Info)
            return True, image_data, metadata, None
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False, None, {}, None
        
        
    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation')
        
        
        img_float = processed_data.astype(float)
        
        # Helper to apply function to each color channel (RGB)
        def apply_per_channel(img, func, *args):
            result = np.zeros_like(img)
            # If color image (3 channels)
            if img.ndim == 3:
                for c in range(3): # Apply to R, G, B
                    result[..., c] = func(img[..., c], *args)
                
                if img.shape[2] == 4:
                    result[..., 3] = img[..., 3]
            else: # Grayscale
                result = func(img, *args)
            return result

        
        # 1. MEDIAN FILTER
        
        if operation == "Median Filter":
            k_size = params.get('kernel_size', 3)
            pad = k_size // 2
            
            def manual_median(channel):
                
                padded = np.pad(channel, ((pad, pad), (pad, pad)), mode='edge')
                views = []
                for i in range(k_size):
                    for j in range(k_size):
                        views.append(padded[i : i + channel.shape[0], j : j + channel.shape[1]])
                
                
                stack = np.stack(views, axis=-1)
                return np.median(stack, axis=-1)

            processed_data = apply_per_channel(img_float, manual_median)

       
        # 2. GAUSSIAN BLUR
   
        elif operation == "Gaussian Blur":
            k_size = params.get('kernel_size', 5)
            sigma = params.get('sigma', 1.0)
            
            
            kernel_1d = self._get_gaussian_1d_kernel(k_size, sigma)
            
            # Applying Horizontal Blur
            
            blur_h = apply_per_channel(img_float, self._apply_convolution, kernel_1d)
            
            # Applying Vertical Blur to the result
            
            processed_data = apply_per_channel(blur_h, self._apply_convolution, kernel_1d.T)
        
        # 3. LAPLACIAN (Edge Detection)
        
        elif operation == "Laplacian (Edge Detect)":
            
            kernel = np.array([
                [ 1,  1,  1],
                [ 1, -8,  1],
                [ 1,  1,  1]
            ])

            raw_edges = apply_per_channel(img_float, self._apply_convolution, kernel)
            
            processed_data = np.abs(raw_edges)

            
            processed_data = processed_data * 2.0
            processed_data = np.clip(processed_data, 0, 255)

            

      
        # 4. POSTERIZATION FILTER
       
        elif operation == "Posterization":
            levels = params.get('levels', 4)
            bucket_size = 255.0 / (levels - 1)
            
            
            processed_data = np.round(img_float / bucket_size) * bucket_size
            
            
            processed_data = np.clip(processed_data, 0, 255)

        
        # 5. SHARPENING (Tunable)
        
        elif operation == "Sharpening":
            strength = params.get('amount', 1.0)
            
           
            # Formula: Identity + (Strength * Laplacian)
            
            
            center = 1.0 + (4.0 * strength)
            
            neg = -strength
            
            kernel = np.array([
                [ 0,   neg,    0],
                [neg, center, neg],
                [ 0,   neg,    0]
            ])
            
            
            processed_data = apply_per_channel(img_float, self._apply_convolution, kernel)
            
            
            processed_data = np.clip(processed_data, 0, 255)

    
        return np.clip(processed_data, 0, 255).astype(np.uint8)