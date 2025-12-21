#!/usr/bin/env python3
"""
Advanced Agentic RAG System - Gradio UI
Unified smart interface with automatic intent detection
"""

import gradio as gr
import os
import asyncio
from pathlib import Path
import shutil
from research_agent import ResearchAssistant
from react_agent import ReActResearchAssistant
from smart_blog_writer import SmartBlogWriter


def detect_intent(query: str) -> str:
    """
    Detect user intent from query.
    
    Returns:
        "blog" if user wants to generate a blog post
        "question" if user wants to ask a question
    """
    blog_keywords = [
        "write", "blog", "create", "generate", "post", "article",
        "draft", "compose", "author", "publish"
    ]
    
    query_lower = query.lower()
    
    # Check for blog keywords
    if any(keyword in query_lower for keyword in blog_keywords):
        return "blog"
    
    return "question"


def get_word_count_from_length(length: str) -> int:
    """Convert length preset to word count."""
    length_map = {
        "Short (300 words)": 300,
        "Medium (500 words)": 500,
        "Long (800 words)": 800
    }
    return length_map.get(length, 500)


def format_react_output(result, show_trace=True):
    """
    Format ReAct agent output with reasoning trace.
    
    Args:
        result: Dictionary from ReAct agent with answer and reasoning_steps
        show_trace: Whether to include reasoning trace
        
    Returns:
        Formatted string with reasoning trace and answer
    """
    output = []
    
    if show_trace and 'reasoning_steps' in result:
        output.append("🧠 REASONING PROCESS")
        output.append("=" * 60)
        output.append("")
        
        for step in result['reasoning_steps']:
            iteration = step['iteration']
            thought = step['thought']
            action = step['action']
            
            output.append(f"Iteration {iteration}:")
            output.append(f"💭 Thought: {thought}")
            output.append(f"🔧 Action: {action}")
            
            if step.get('observation'):
                obs = step['observation']
                # Truncate long observations
                if len(obs) > 300:
                    obs = obs[:300] + "..."
                output.append(f"👁️  Observation: {obs}")
            
            output.append("")
        
        output.append("=" * 60)
        output.append("")
    
    # Add final answer
    output.append("✅ FINAL ANSWER")
    output.append("=" * 60)
    output.append("")
    output.append(result['answer'])
    output.append("")
    
    # Add statistics
    if 'total_iterations' in result:
        output.append("=" * 60)
        output.append("📊 STATISTICS")
        output.append(f"• Reasoning steps: {result['total_iterations']}")
        
        if 'reasoning_steps' in result:
            tools_used = len([s for s in result['reasoning_steps'] if s['action'] != 'FINISH'])
            output.append(f"• Tools used: {tools_used}")
    
    return "\n".join(output)


def process_request(
    download_arxiv,
    arxiv_topic,
    num_papers,
    load_existing_papers,
    uploaded_file,
    user_query,
    output_style,
    output_length,
    max_iterations,
    show_reasoning,
    progress=gr.Progress()
):
    """
    Main processing function with automatic intent detection.
    """
    try:
        # Validate inputs
        if not user_query or user_query.strip() == "":
            return "❌ Error: Please enter your question or request.", None
        
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "❌ Error: OPENAI_API_KEY environment variable not set.", None
        
        # Detect intent
        intent = detect_intent(user_query)
        print(f"\n🎯 Detected Intent: {intent.upper()}")
        
        progress(0.1, desc="Analyzing your request...")
        
        # Convert length to word count
        word_count = get_word_count_from_length(output_length)
        
        # Map UI style to backend style
        style_map = {
            "Technical/Academic": "technical",
            "Professional/Business": "professional",
            "Casual/Conversational": "casual"
        }
        style = style_map.get(output_style, "professional")
        
        if intent == "blog":
            # BLOG GENERATION MODE
            progress(0.2, desc="Initializing Smart Blog Writer...")
            
            # Create smart blog writer
            smart_writer = SmartBlogWriter(openai_api_key=api_key)
            
            progress(0.3, desc="Agent deciding information sources...")
            
            # Use smart blog writer - it will decide sources automatically
            blog_content, metadata = asyncio.run(smart_writer.write_smart_blog(
                user_query=user_query,
                download_enabled=download_arxiv,
                num_papers_to_download=int(num_papers) if download_arxiv else 3,
                style=style,
                word_count=word_count
            ))
            
            progress(1.0, desc="Done!")
            
            # Format output with metadata
            output = []
            output.append("🧠 SMART BLOG WRITER")
            output.append("=" * 60)
            output.append(f"📝 Your Request: {metadata['query']}")
            output.append(f"🎯 Detected Topic: {metadata['topic']}")
            output.append(f"📊 Information Source: {metadata['source_type'].upper()}")
            
            if metadata['papers_used']:
                output.append(f"📚 Papers Used ({len(metadata['papers_used'])}):")
                for paper in metadata['papers_used']:
                    output.append(f"   • {paper}")
            
            output.append("=" * 60)
            output.append("")
            output.append("📝 GENERATED BLOG POST")
            output.append("=" * 60)
            output.append("")
            output.append(blog_content)
            
            result = "\n".join(output)
            
            # Save to file
            os.makedirs("./blog_posts", exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d")
            safe_topic = metadata['topic'].replace(" ", "_")[:50]
            filename = f"./blog_posts/blog_{timestamp}_{safe_topic}.txt"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(blog_content)
            
            print(f"✅ Blog saved to: {filename}")
            
            return result, blog_content
        
        else:
            # QUESTION ANSWERING MODE
            progress(0.2, desc="Gathering papers...")
            
            # Collect papers
            papers = []
            
            # 1. Load existing papers from datasets folder if checked
            if load_existing_papers:
                existing_papers = [str(p) for p in Path("./datasets").glob("*.pdf")]
                papers.extend(existing_papers)
                progress(0.25, desc=f"Loaded {len(existing_papers)} existing papers...")
            
            # 2. Handle arXiv download
            if download_arxiv:
                if not arxiv_topic or arxiv_topic.strip() == "":
                    return "❌ Error: Please enter a topic for arXiv papers.", None
                
                progress(0.3, desc=f"Downloading {num_papers} papers from arXiv...")
                
                from arxiv_downloader import download_arxiv_papers
                downloaded = download_arxiv_papers(
                    topic=arxiv_topic,
                    max_results=num_papers,
                    download_dir="./datasets",
                    months_back=12
                )
                papers.extend(downloaded)
            
            # Handle uploaded file
            if uploaded_file is not None:
                progress(0.4, desc="Processing uploaded file...")
                
                # Save uploaded file to datasets folder
                upload_path = Path("./datasets") / Path(uploaded_file.name).name
                shutil.copy(uploaded_file.name, upload_path)
                papers.append(str(upload_path))
            
            # Check if we have papers
            if not papers:
                return "❌ Error: No papers provided. Please enable paper sources or upload a PDF.", None
            
            progress(0.5, desc=f"Initializing Advanced Agentic RAG with {len(papers)} papers...")
            
            # Create base Research Assistant
            base_assistant = ResearchAssistant(papers=papers)
            base_assistant.setup()
            
            # Wrap with ReAct for autonomous reasoning
            progress(0.6, desc="Setting up ReAct agent for multi-step reasoning...")
            assistant = ReActResearchAssistant(
                base_assistant=base_assistant,
                max_iterations=int(max_iterations),
                verbose=show_reasoning
            )
            assistant.setup()
            
            progress(0.7, desc="Processing your question with multi-step reasoning...")
            
            # Query using ReAct agent
            result = asyncio.run(assistant.query(user_query))
            
            progress(1.0, desc="Done!")
            
            # Format output with reasoning trace
            formatted_output = format_react_output(result, show_trace=show_reasoning)
            
            return formatted_output, None
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, None


def create_ui():
    """
    Create the unified smart Gradio interface.
    """
    with gr.Blocks(title="Advanced Agentic RAG") as app:
        gr.Markdown("""
        # 🧠 Advanced Agentic RAG System
        
        **Intelligent multi-step reasoning over research papers**
        
        Simply describe what you want - the agent will figure out the rest!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📚 Paper Sources (Optional)")
                
                download_arxiv = gr.Checkbox(
                    label="Auto-download papers from arXiv if needed",
                    value=False,
                    info="Agent will download relevant papers when none are found"
                )
                
                arxiv_topic = gr.Textbox(
                    label="Topic for arXiv papers",
                    placeholder="e.g., computer vision, transformers, RAG systems",
                    visible=False
                )
                
                num_papers = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of papers to download",
                    visible=False
                )
                
                load_existing_papers = gr.Checkbox(
                    label="Include all papers from datasets folder",
                    value=False,
                    info="Use existing PDFs in your datasets directory"
                )
                
                uploaded_file = gr.File(
                    label="📎 Or upload a PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                gr.Markdown("---")
                gr.Markdown("### 💬 What would you like to do?")
                
                user_query = gr.Textbox(
                    label="",
                    placeholder="""Examples:
• "Write a blog on computer vision applications"
• "What are the main approaches in RAG systems?"
• "Compare autonomous agents across papers"
• "Create a technical article about transformers"
• "Explain the key concepts in this research"
""",
                    lines=4
                )
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ Output Settings")
                
                output_style = gr.Dropdown(
                    choices=["Technical/Academic", "Professional/Business", "Casual/Conversational"],
                    value="Professional/Business",
                    label="Writing Style",
                    info="Applies to both blogs and answers"
                )
                
                output_length = gr.Dropdown(
                    choices=["Short (300 words)", "Medium (500 words)", "Long (800 words)"],
                    value="Medium (500 words)",
                    label="Output Length",
                    info="Target length for generated content"
                )
                
                gr.Markdown("### 🧠 Advanced Reasoning")
                
                max_iterations = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Max Reasoning Steps",
                    info="How many reasoning iterations the agent can perform"
                )
                
                show_reasoning = gr.Checkbox(
                    label="Show detailed reasoning trace",
                    value=True,
                    info="Display the agent's thought process step-by-step"
                )
                
                gr.Markdown("---")
                
                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                    clear_btn = gr.ClearButton(size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Output")
                
                output_text = gr.Textbox(
                    label="Generated Content",
                    lines=25,
                    max_lines=35
                )
                
                download_btn = gr.DownloadButton(
                    label="💾 Download as .txt",
                    visible=False
                )
        
        # Show/hide controls based on selections
        def toggle_arxiv_controls(download):
            return {
                arxiv_topic: gr.update(visible=download),
                num_papers: gr.update(visible=download)
            }
        
        download_arxiv.change(
            fn=toggle_arxiv_controls,
            inputs=[download_arxiv],
            outputs=[arxiv_topic, num_papers]
        )
        
        # Generate button click
        def generate_and_show_download(
            download_arxiv, arxiv_topic, num_papers, load_existing_papers, uploaded_file,
            user_query, output_style, output_length, max_iterations, show_reasoning
        ):
            result, download_content = process_request(
                download_arxiv, arxiv_topic, num_papers, load_existing_papers, uploaded_file,
                user_query, output_style, output_length, max_iterations, show_reasoning
            )
            
            # Show download button if content is available
            show_download = download_content is not None
            
            if show_download:
                # Save to temp file for download
                temp_file = f"./blog_posts/temp_blog.txt"
                os.makedirs("./blog_posts", exist_ok=True)
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(download_content)
                
                return result, gr.update(visible=True, value=temp_file)
            else:
                return result, gr.update(visible=False)
        
        generate_btn.click(
            fn=generate_and_show_download,
            inputs=[
                download_arxiv, arxiv_topic, num_papers, load_existing_papers, uploaded_file,
                user_query, output_style, output_length, max_iterations, show_reasoning
            ],
            outputs=[output_text, download_btn]
        )
        
        # Clear button
        clear_btn.add([
            arxiv_topic, num_papers, uploaded_file, user_query, output_text
        ])
        
        gr.Markdown("""
        ---
        ### 💡 About This System
        
        This **Advanced Agentic RAG** system automatically detects your intent and responds accordingly:
        
        **🤖 Automatic Intent Detection:**
        - Detects if you want a **blog post** or an **answer** to a question
        - No need to select modes - just describe what you want!
        
        **🧠 Smart Blog Writer:**
        - Extracts topic from your request
        - Searches for relevant papers automatically
        - Downloads from arXiv if needed
        - Generates comprehensive blog posts
        
        **🔍 ReAct Agent for Q&A:**
        - Multi-step autonomous reasoning
        - Self-directed tool selection
        - Transparent decision-making
        - Complex query handling
        
        **💡 Tips:**
        - For blogs: Start with "Write a blog on..." or "Create an article about..."
        - For questions: Ask naturally like "What are..." or "Explain..."
        - Enable paper sources for better results
        - Adjust reasoning steps for complex queries
        
        **Examples:**
        ```
        Blog: "Write a blog on computer vision applications"
        Q&A:  "What are the main approaches in RAG systems?"
        Blog: "Create a technical article about transformers"
        Q&A:  "Compare autonomous agents across these papers"
        ```
        """)
    
    return app


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set it before using the app:")
        print("export OPENAI_API_KEY='your-key-here'")
        print()
    
    print("🧠 Launching Advanced Agentic RAG System...")
    print("=" * 60)
    print("Features:")
    print("  ✅ Automatic intent detection (blog vs question)")
    print("  ✅ Smart blog writer with source selection")
    print("  ✅ Multi-step autonomous reasoning (ReAct)")
    print("  ✅ Unified, modern interface")
    print("=" * 60)
    print()
    
    # Create and launch app
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
    )
