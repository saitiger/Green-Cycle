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

#B2B Solution
raw_material_template = """You are an expert in Life-cycle assessment. You have been hired as a Life-cycle assessment consultant, to perform Life-cycle assessment of a product.
You have been given a product. The product name is given below:
***
# PRODUCT NAME: {product_name}
***
Analyse, Brainstorm the Identify all the Raw Materials that are involved in manufacturing this product.
{format_instructions}
"""

process_template = """You are an expert in Manufacturing and Industrial Engineering.
Given below is a list raw materials.
***
# Raw Materials: 
{raw_material}
***
Analyse and tell the manufacturing process involved in making each one of the Raw Materials. If you know the exact manufacturing process terminology then say it. If no process is involved and it is naturally occurring then just say no processing required.
Give the raw material alongside it's process that you identify on each new line, in the format "raw material" "==>" "manufacturing process"
"""

analysis_template = """You have done P.H.D in Product Lifecycle Assessment Analysis. And are now working as a sustainability consultant.
Given below is the product name. The list of raw materials required in manufacturing it and manufacturing processes involved for each raw material.
***
# Product: {product}
***
# Raw Material and Manufacturing Processes: {details}
***
Analyse the details and give top 5 recommendations on how to reduce the emissions across the supply chain for this product. Focus on newer green solutions that can be used.
Give recommendations in bullet points.
"""

# Vision-Based Templates for B2C Applications
product_vision_template = """You are a computer vision expert specialized in product identification.
Analyze the provided image and identify the main product visible in it.
Focus on the primary product and its material composition.
Return just the product name without any additional explanation."""

quick_sustainability_template = """You are a sustainability expert. Given a product, provide detailed sustainability metrics:

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

# Initialize all chains
material_chain = LLMChain(
    llm=llm, 
    prompt=PromptTemplate(
        template=raw_material_template,
        input_variables=["product_name"],
        partial_variables={"format_instructions": format_instructions}
    ),
    verbose=True
)

process_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=process_template,
        input_variables=["raw_material"]
    ),
    verbose=True
)

analysis_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=analysis_template,
        input_variables=["product", "details"]
    ),
    verbose=True
)

product_vision_chain = LLMChain(
    llm=vision_model,
    prompt=PromptTemplate(
        template=product_vision_template,
        input_variables=["image"]
    ),
    verbose=True
)

quick_sustainability_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=quick_sustainability_template,
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
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        product_name = product_vision_chain.predict(image=image_data)
        return product_name.strip()
    except Exception as e:
        return f"Error processing image: {str(e)}"

def get_materials_from_gpt(product_name):
    """Get raw materials for a product"""
    materials = material_chain.predict(product_name=product_name)
    return output_parser.parse(materials)

def get_processes(raw_material):
    """Get manufacturing processes for raw materials"""
    return process_chain.predict(raw_material=raw_material)

def get_analysis(product, details):
    """Get LCA analysis and recommendations"""
    return analysis_chain.predict(product=product, details=details)

def get_quick_sustainability(product_name):
    """Get quick sustainability metrics"""
    return quick_sustainability_chain.predict(product_name=product_name)

def get_diy_ideas(product_name):
    """Get DIY ideas for the product"""
    return diy_chain.predict(product_name=product_name)

with gr.Blocks() as demo:
    gr.Markdown("# Comprehensive LCA Assistant")
    
    with gr.Tabs():
        # Quick Scan Tab
        with gr.Tab("Quick Scan"):
            with gr.Row():
                with gr.Column():
                    camera_input = gr.Image(source="webcam", type="pil", label="Scan Product")
                    quick_manual_input = gr.Textbox(label="Or Enter Product Name Manually")
                    quick_scan_button = gr.Button("Quick Scan")
                
                with gr.Column():
                    product_name_display = gr.Markdown(label="Identified Product")
                    sustainability_metrics = gr.Markdown(label="Quick Sustainability Metrics")
                    diy_suggestions = gr.Markdown(label="DIY Ideas")

        # Detailed LCA Tab
        with gr.Tab("Detailed LCA"):
            with gr.Column():
                product_name = gr.Textbox(label="Enter Product for LCA Analysis")
                tell_me_materials = gr.Button("1. Identify Raw Materials")
                raw_materials = gr.Markdown()
                
                get_processes_btn = gr.Button("2. Identify Processes Involved")
                processes = gr.Markdown()
                
                get_analysis_btn = gr.Button("3. Analysis and Recommendations")
                analysis = gr.Markdown()

    def quick_analyze_product(image, manual_name):
        """Handle quick scan analysis"""
        if image is not None:
            product_name = process_image(image)
        else:
            product_name = manual_name
        
        if not product_name:
            return "No product identified", "", ""
        
        metrics = get_quick_sustainability(product_name)
        diy_ideas = get_diy_ideas(product_name)
        
        return (
            f"# Identified Product: {product_name}",
            metrics,
            diy_ideas
        )

    def identify_raw_materials(product_name):
        """Handle raw materials identification"""
        response = get_materials_from_gpt(product_name)
        result = "# Raw Materials\n"
        for i in response:
            result += "- " + i + "\n"
        return result
    
    def identify_processes(raw_materials):
        """Handle process identification"""
        result = "# Processes:\n"
        response = get_processes(raw_materials)
        return response
    
    def identify_analysis(product_name, processes):
        """Handle detailed analysis"""
        result = get_analysis(product_name, processes)
        return result

    quick_scan_button.click(
        fn=quick_analyze_product,
        inputs=[camera_input, quick_manual_input],
        outputs=[product_name_display, sustainability_metrics, diy_suggestions]
    )

    tell_me_materials.click(
        fn=identify_raw_materials,
        inputs=product_name,
        outputs=raw_materials
    )

    get_processes_btn.click(
        fn=identify_processes,
        inputs=raw_materials,
        outputs=processes
    )

    get_analysis_btn.click(
        fn=identify_analysis,
        inputs=[product_name, processes],
        outputs=analysis
    )

demo.launch()
