from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel
from dotenv import load_dotenv
import gradio as gr
import openai
from PIL import Image
import io

load_dotenv()

# Output parser setup
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

# Enhanced prompt templates
product_vision_template = """You are a computer vision expert specialized in product identification.
Analyze the provided image and identify the main product visible in it.
Focus on the primary product and its material composition.
Return just the product name without any additional explanation."""

sustainability_template = """You are a sustainability expert. Given a product, provide detailed sustainability metrics:

Product: {product_name}

Provide the following metrics:
1. Estimated carbon footprint (in CO2e)
2. Water usage in production (in liters)
3. Packaging material analysis
4. Reusability score (1-10)
5. Recycling recommendations

Format the response in markdown."""

diy_template = """You are a creative DIY expert focused on sustainability.
Product: {product_name}

Suggest 3 creative DIY ideas to:
1. Reuse or upcycle this product
2. Reduce its environmental impact
3. Extend its lifecycle

Focus on practical, achievable ideas that don't require specialized tools.
Format the response in markdown with bullet points."""

# LLM Chain setup
llm = OpenAI(temperature=0.5, model_name="gpt-4")
vision_model = OpenAI(temperature=0.2, model_name="gpt-4-vision-preview")

# Initialize chains
product_vision_chain = LLMChain(
    llm=vision_model,
    prompt=PromptTemplate(
        template=product_vision_template,
        input_variables=["image"]
    ),
    verbose=True
)

sustainability_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=sustainability_template,
        input_variables=["product_name"]
    ),
    verbose=True
)

diy_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=diy_template,
        input_variables=["product_name"]
    ),
    verbose=True
)

def process_image(image):
    """Process the captured image and identify the product"""
    try:
        # Convert image to base64 or appropriate format for vision model
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        # Get product identification from vision model
        product_name = product_vision_chain.predict(image=image_data)
        return product_name.strip()
    except Exception as e:
        return f"Error processing image: {str(e)}"

def get_sustainability_metrics(product_name):
    """Get sustainability metrics for the identified product"""
    return sustainability_chain.predict(product_name=product_name)

def get_diy_ideas(product_name):
    """Get DIY ideas for the identified product"""
    return diy_chain.predict(product_name=product_name)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Sustainability Assistant with Vision")
    
    with gr.Row():
        with gr.Column():
            # Camera input
            camera_input = gr.Image(source="webcam", type="pil", label="Scan Product")
            manual_input = gr.Textbox(label="Or Enter Product Name Manually")
            scan_button = gr.Button("Scan and Analyze")
        
        with gr.Column():
            # Output displays
            product_name_display = gr.Markdown(label="Identified Product")
            sustainability_metrics = gr.Markdown(label="Sustainability Metrics")
            diy_suggestions = gr.Markdown(label="DIY Ideas")

    def analyze_product(image, manual_name):
        """Main function to process input and generate analysis"""
        if image is not None:
            product_name = process_image(image)
        else:
            product_name = manual_name
        
        if not product_name:
            return "No product identified", "", ""
        
        # Get analysis
        metrics = get_sustainability_metrics(product_name)
        diy_ideas = get_diy_ideas(product_name)
        
        return (
            f"# Identified Product: {product_name}",
            metrics,
            diy_ideas
        )

    # Event handlers
    scan_button.click(
        fn=analyze_product,
        inputs=[camera_input, manual_input],
        outputs=[product_name_display, sustainability_metrics, diy_suggestions]
    )

demo.launch()
