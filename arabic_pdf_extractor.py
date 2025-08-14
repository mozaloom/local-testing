#!/usr/bin/env python3
"""
Arabic PDF Text Extraction using OCR VLM (Visual Language Model)
Extracts Arabic text from PDF files using Ollama OCR models with Arabic language support
"""

import os
import json
import base64
import time
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF for PDF processing
import requests
from PIL import Image, ImageEnhance
import io
from typing import List, Dict, Any
import unicodedata

class ArabicPDFOCRExtractor:
    def __init__(self, model_name="moondream:latest",
                 ollama_url="http://localhost:11434"):
        """
        Initialize the Arabic PDF OCR Extractor
        
        Args:
            model_name: Name of the OCR VLM model in Ollama (optimized for 6GB VRAM)
            ollama_url: URL of the Ollama API endpoint
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/generate"
        
        # GPU-friendly models for 6GB VRAM (ordered by preference based on your setup)
        self.fallback_models = [
            "moondream:latest",                                              # 1.7GB - Fast, good for basic OCR
            "hf.co/mradermacher/OCR_VLM-Qwen2.5VL-3B-final-GGUF:Q4_K_M",   # 2.8GB - Better OCR quality
            "hf.co/mradermacher/OCR_VLM-Qwen2.5VL-3B-final-GGUF:Q8_0",     # 4.1GB - Best OCR quality  
            "bakllava:latest"                                                # 4.7GB - Good general vision
        ]
        
    def check_available_models(self) -> List[str]:
        """
        Check which vision models are available in Ollama
        
        Returns:
            List of available vision model names
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            
            # Filter for vision-capable models
            vision_models = []
            for model in available_models:
                if any(keyword in model.lower() for keyword in ['vision', 'llava', 'moondream', 'bakllava', 'ocr']):
                    vision_models.append(model)
                    
            print(f"Available vision models: {vision_models}")
            return vision_models
            
        except Exception as e:
            print(f"Error checking models: {e}")
            return []
        
    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF pages to images with optimization for Arabic text
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image conversion (higher for better Arabic OCR)
            
        Returns:
            List of PIL Images, one per page
        """
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Use higher DPI for Arabic text clarity
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat, alpha=False)  # Remove alpha channel
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image for better OCR
            image = self.enhance_image_for_ocr(image)
            
            # Resize if too large but maintain quality for Arabic text
            max_size = 2560  # Higher than before for Arabic clarity
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            images.append(image)
            
        doc.close()
        return images
    
    def enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better Arabic OCR results
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
        
    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string with optimized compression
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string of the image
        """
        buffered = io.BytesIO()
        
        # Use PNG for better quality with Arabic text
        image.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def clean_arabic_text(self, text: str) -> str:
        """
        Clean and normalize Arabic text output
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned Arabic text
        """
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Normalize Arabic text
        text = unicodedata.normalize('NFKC', text)
        
        # Remove common OCR artifacts
        artifacts = ['忘记了', '古怪', '焦急', '狰', '龇', '嚯', '哐', '邋', '唏', '轱', '尬']
        for artifact in artifacts:
            text = text.replace(artifact, '')
        
        # Clean up mixed language artifacts
        import re
        # Remove standalone numbers and symbols that are clearly OCR errors
        text = re.sub(r'\b\d+[a-zA-Z]*\d*\b(?=\s|$)', '', text)
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\d\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Clean excessive spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def extract_text_from_image(self, image_base64: str, max_retries: int = 3) -> str:
        """
        Extract Arabic text from image using OCR VLM model via Ollama
        
        Args:
            image_base64: Base64 encoded image
            max_retries: Maximum number of retry attempts
            
        Returns:
            Extracted Arabic text from the image
        """
        
        # Enhanced prompt specifically for Arabic OCR
        prompt = """You are an expert Arabic OCR system. Extract ALL Arabic text from this image with perfect accuracy.

Instructions:
1. Focus ONLY on Arabic text - ignore any other languages
2. Maintain the original text structure and formatting
3. Include all visible Arabic words, numbers, and punctuation
4. Preserve line breaks and paragraph structure
5. Do not translate - provide the original Arabic text only
6. If you see legal document headers, titles, or article numbers, include them
7. Ignore watermarks, page numbers, or decorative elements

Please provide a clean, accurate transcription of all Arabic text in the image."""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.0,  # Most deterministic
                "top_p": 0.1,
                "num_ctx": 8192,
                "num_predict": 4096
            }
        }
        
        for attempt in range(max_retries):
            try:
                print(f"    Attempt {attempt + 1}/{max_retries} with model {self.model_name}")
                response = requests.post(self.api_url, json=payload, timeout=180)
                
                if response.status_code == 500:
                    print(f"    Server error (500), retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                    continue
                elif response.status_code == 404:
                    print(f"    Model not found, trying fallback models...")
                    return self.try_fallback_models(image_base64)
                    
                response.raise_for_status()
                result = response.json()
                extracted_text = result.get("response", "").strip()
                
                # Clean the extracted text
                cleaned_text = self.clean_arabic_text(extracted_text)
                
                if cleaned_text:
                    # Check if we got meaningful Arabic content
                    arabic_chars = sum(1 for c in cleaned_text if '\u0600' <= c <= '\u06FF')
                    if arabic_chars > 10:  # Minimum threshold for Arabic content
                        return cleaned_text
                    else:
                        print(f"    Insufficient Arabic content, retrying...")
                        continue
                else:
                    print(f"    Empty response after cleaning, retrying...")
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"    Timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                print(f"    API Error on attempt {attempt + 1}: {e}")
            except json.JSONDecodeError as e:
                print(f"    JSON Error on attempt {attempt + 1}: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(2)
                
        print(f"    Failed after {max_retries} attempts")
        return ""
    
    def try_fallback_models(self, image_base64: str) -> str:
        """
        Try fallback models if primary model fails
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Extracted text using fallback model
        """
        available_models = self.check_available_models()
        
        for fallback_model in self.fallback_models:
            if fallback_model in available_models:
                print(f"    Trying fallback model: {fallback_model}")
                original_model = self.model_name
                self.model_name = fallback_model
                
                try:
                    result = self.extract_text_from_image(image_base64, max_retries=1)
                    if result:
                        print(f"    Success with fallback model: {fallback_model}")
                        return result
                except:
                    continue
                finally:
                    self.model_name = original_model
                    
        return ""
            
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single Arabic PDF file and extract text from all pages
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted Arabic text and metadata
        """
        print(f"Processing Arabic PDF: {pdf_path}")
        
        # Convert PDF to images
        try:
            images = self.pdf_to_images(pdf_path)
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return {"error": f"Failed to convert PDF: {str(e)}"}
            
        # Extract text from each page
        pages_text = []
        total_pages = len(images)
        
        for page_num, image in enumerate(images, 1):
            print(f"  Processing page {page_num}/{total_pages}")
            
            try:
                # Convert image to base64
                image_base64 = self.image_to_base64(image)
                
                # Check image size
                image_size_mb = len(image_base64) * 3 / 4 / (1024 * 1024)
                print(f"    Image size: ~{image_size_mb:.1f}MB")
                
                if image_size_mb > 20:  # Increased limit for Arabic text quality
                    print(f"    Image too large, compressing...")
                    # Compress the image more
                    compressed_image = image.copy()
                    compressed_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    image_base64 = self.image_to_base64(compressed_image)
                
                # Extract Arabic text using OCR VLM
                extracted_text = self.extract_text_from_image(image_base64)
                
                page_data = {
                    "page_number": page_num,
                    "text": extracted_text,
                    "character_count": len(extracted_text),
                    "arabic_char_count": sum(1 for c in extracted_text if '\u0600' <= c <= '\u06FF')
                }
                    
            except Exception as e:
                print(f"    Error processing page {page_num}: {e}")
                page_data = {
                    "page_number": page_num,
                    "text": f"[ERROR: {str(e)}]",
                    "character_count": 0,
                    "arabic_char_count": 0,
                    "error": str(e)
                }
                
            pages_text.append(page_data)
            
        # Combine all text
        full_text = "\n\n".join([page["text"] for page in pages_text if page["text"] and not page["text"].startswith("[ERROR")])
        total_arabic_chars = sum(page.get("arabic_char_count", 0) for page in pages_text)
        
        # Create structured result
        result = {
            "source_file": os.path.basename(pdf_path),
            "source_path": pdf_path,
            "extraction_timestamp": datetime.now().isoformat(),
            "model_used": self.model_name,
            "total_pages": total_pages,
            "total_characters": len(full_text),
            "total_arabic_characters": total_arabic_chars,
            "pages": pages_text,
            "full_text": full_text,
            "metadata": {
                "extraction_method": "Arabic_OCR_VLM",
                "has_arabic_content": total_arabic_chars > 50,
                "arabic_content_percentage": (total_arabic_chars / len(full_text) * 100) if full_text else 0,
                "average_chars_per_page": len(full_text) / total_pages if total_pages > 0 else 0,
                "successful_pages": len([p for p in pages_text if not p.get("error")])
            }
        }
        
        return result
        
    def process_directory(self, input_dir: str, output_dir: str):
        """
        Process all Arabic PDF files in a directory and subdirectories
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save JSON results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(input_path.rglob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return
            
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        processed_count = 0
        failed_count = 0
        total_arabic_chars = 0
        
        for pdf_file in pdf_files:
            try:
                # Process the PDF
                result = self.process_pdf(str(pdf_file))
                
                # Create output filename
                relative_path = pdf_file.relative_to(input_path)
                output_file = output_path / (str(relative_path).replace(".pdf", "_arabic.json"))
                
                # Create subdirectories if needed
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save result as JSON with Arabic support
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                if "error" in result:
                    print(f"  ❌ Failed: {result['error']}")
                    failed_count += 1
                else:
                    arabic_chars = result['metadata'].get('total_arabic_characters', 0)
                    total_arabic_chars += arabic_chars
                    print(f"  ✅ Saved: {output_file} (Arabic chars: {arabic_chars})")
                    processed_count += 1
                    
            except Exception as e:
                print(f"  ❌ Error processing {pdf_file}: {e}")
                failed_count += 1
                
        print(f"\n📊 Arabic PDF Processing Complete:")
        print(f"   ✅ Successfully processed: {processed_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   🔤 Total Arabic characters extracted: {total_arabic_chars:,}")
        print(f"   📁 Results saved to: {output_path}")

def main():
    """Main function to run the Arabic PDF OCR extraction"""
    
    # Configuration
    input_directory = "data"
    output_directory = "extracted_arabic_text"
    
    print("🇸🇦 Arabic PDF OCR Text Extraction with VLM")
    print("=" * 60)
    
    # Check if input directory exists
    if not os.path.exists(input_directory):
        print(f"❌ Input directory '{input_directory}' not found!")
        return
        
    # Initialize extractor with best OCR model that fits your GPU
    extractor = ArabicPDFOCRExtractor(
        model_name="moondream:latest"
    )
    
    # Test Ollama connection and check available models
    try:
        response = requests.get(f"{extractor.ollama_url}/api/tags", timeout=10)
        response.raise_for_status()
        print("✅ Ollama connection successful")
        
        # Check available vision models
        available_models = extractor.check_available_models()
        if not available_models:
            print("⚠️  No vision models found. Installing GPU-friendly model...")
            print("For RTX 3060 6GB, run: ollama pull moondream")
            print("Alternative: ollama pull bakllava")
            return
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Please make sure Ollama is running with: ollama serve")
        return
        
    # Process all Arabic PDFs
    extractor.process_directory(input_directory, output_directory)

if __name__ == "__main__":
    main()