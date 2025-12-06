"""
NeuroSound Demo - Interactive Streamlit App

Usage:
    streamlit run demo.py
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

try:
    from neurosound import NeuroSound, NeuroSoundUniversal
except ImportError:
    st.error("⚠️ NeuroSound not installed. Run: pip install neurosound")
    st.stop()


def main():
    st.set_page_config(
        page_title="NeuroSound Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 NeuroSound - Content-Aware Audio Compression")
    st.markdown("**15-25x typical compression with intelligent optimizations**")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        version = st.radio(
            "Version",
            options=['v3.2 UNIVERSAL', 'v3.1 Classic'],
            index=0,
            help="""
            - **v3.2 UNIVERSAL:** Multi-format (MP3/AAC/OGG/FLAC/WAV), 80.94x ratio
            - **v3.1 Classic:** WAV only, 12.52x ratio (faster)
            """
        )
        
        mode = st.selectbox(
            "Compression Mode",
            options=['balanced', 'aggressive', 'safe'],
            index=0,
            help="""
            - **Balanced:** Optimal ratio (RECOMMENDED)
            - **Aggressive:** Fastest processing
            - **Safe:** Highest quality
            """
        )
        
        st.markdown("---")
        
        if version == 'v3.2 UNIVERSAL':
            st.markdown("""
            ### 📊 v3.2 UNIVERSAL
            
            **4 Content-Aware Techniques:**
            - Silence detection/removal
            - Stereo→Mono (98% threshold)
            - Adaptive normalization
            - Multi-resolution FFT
            
            **Formats:**
            - MP3, AAC, OGG
            - FLAC, WAV, M4A
            
            **Performance:**
            - Typical: 15-25x
            - Best: 30-50x (silence-heavy)
            """)
        else:
            st.markdown("""
            ### 📊 v3.1 Classic
            
            **Spectral Analysis:**
            - FFT peak detection
            - Adaptive VBR selection
            
            **Formats:**
            - WAV only
            
            **Performance:**
            - Ratio: 12.52x
            - Time: 0.105s
            - Energy: 29mJ
            """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload")
        
        if version == 'v3.2 UNIVERSAL':
            file_types = ['wav', 'mp3', 'aac', 'ogg', 'flac', 'm4a']
            help_text = "Upload any audio file (MP3, AAC, OGG, FLAC, WAV, M4A)"
        else:
            file_types = ['wav']
            help_text = "Upload a 16-bit PCM WAV file"
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=file_types,
            help=help_text
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size = len(uploaded_file.getvalue())
            file_ext = Path(uploaded_file.name).suffix.lower()
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.info(f"📦 Original size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            
            # Play original (format detection)
            if file_ext == '.wav':
                st.audio(uploaded_file, format='audio/wav')
            elif file_ext == '.mp3':
                st.audio(uploaded_file, format='audio/mp3')
            elif file_ext in ['.aac', '.m4a']:
                st.audio(uploaded_file, format='audio/mp4')
            elif file_ext == '.ogg':
                st.audio(uploaded_file, format='audio/ogg')
            elif file_ext == '.flac':
                st.audio(uploaded_file, format='audio/flac')
    
    with col2:
        st.header("📥 Compressed")
        
        if uploaded_file is not None:
            if st.button("🚀 Compress", type="primary"):
                with st.spinner("Compressing..."):
                    # Create temp files
                    file_ext = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_input:
                        tmp_input.write(uploaded_file.getvalue())
                        tmp_input_path = tmp_input.name
                    
                    tmp_mp3_path = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
                    
                    try:
                        # Select codec version
                        if version == 'v3.2 UNIVERSAL':
                            codec = NeuroSoundUniversal(mode=mode)
                        else:
                            codec = NeuroSound(mode=mode)
                        
                        # Compress
                        size, ratio, energy = codec.compress(
                            tmp_input_path,
                            tmp_mp3_path,
                            verbose=False
                        )
                        
                        # Display results
                        st.success("✅ Compression complete!")
                        
                        # Metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                "Compression Ratio",
                                f"{ratio:.2f}x",
                                delta=f"{100*(1-1/ratio):.1f}% smaller"
                            )
                        
                        with metric_col2:
                            st.metric(
                                "Compressed Size",
                                f"{size:,} bytes",
                                delta=f"{size/1024:.1f} KB"
                            )
                        
                        with metric_col3:
                            if version == 'v3.2 UNIVERSAL':
                                st.metric(
                                    "Energy",
                                    f"{energy:.0f} mJ",
                                    delta="4 innovations"
                                )
                            else:
                                st.metric(
                                    "Energy",
                                    f"{energy:.0f} mJ",
                                    delta=f"{((47-energy)/47*100):.0f}% saved vs baseline"
                                )
                        
                        # Read compressed file
                        with open(tmp_mp3_path, 'rb') as f:
                            mp3_data = f.read()
                        
                        # Play compressed
                        st.audio(mp3_data, format='audio/mp3')
                        
                        # Download button
                        output_name = Path(uploaded_file.name).stem + '_compressed.mp3'
                        st.download_button(
                            "⬇️ Download MP3",
                            data=mp3_data,
                            file_name=output_name,
                            mime='audio/mp3'
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                    
                    finally:
                        # Cleanup
                        os.remove(tmp_input_path)
                        if os.path.exists(tmp_mp3_path):
                            os.remove(tmp_mp3_path)
        else:
            st.info("👆 Upload an audio file to begin")
    
    # Footer
    st.markdown("---")
    
    st.markdown("""
    ### 🔬 How It Works
    
    **v3.2 UNIVERSAL** uses 4 content-aware techniques:
    
    1. **Psychoacoustic Silence Detection:** Remove < -50dB sections (effective on podcast/voix)
    2. **Intelligent Stereo→Mono:** 98% correlation threshold (works on quasi-mono content)
    3. **Adaptive Normalization:** -1dB headroom for optimal VBR encoding (always beneficial)
    4. **Multi-Resolution FFT:** Hybrid 50ms + 1s tonality analysis (improved from v3.1)
    
    **v3.1 Classic** uses spectral content analysis:
    
    1. **FFT Peak Detection:** Analyze audio tonality (pure tone vs complex music)
    2. **Adaptive VBR:** Select optimal MP3 VBR setting based on content
    3. **Smart Optimization:** DC offset removal, joint stereo for correlated L/R
    
    **Result:** 15-25x typical (v3.2), 30-50x on silence-heavy audio, 12.52x proven (v3.1). Perceptual transparency maintained.
    
    *Note: Compression ratio varies with content. Simple/sparse audio benefits most from v3.2 optimizations.*
    
    ### 📚 Learn More
    
    - [GitHub Repository](https://github.com/bhanquier/neuroSound)
    - [PyPI Package](https://pypi.org/project/neurosound/)
    - [Technical Article](https://github.com/bhanquier/neuroSound/blob/main/ARTICLE.md)
    
    ### 💚 Environmental Impact
    
    If adopted globally:
    - 💡 38.5 TWh saved/year
    - 🌱 19M tons CO₂ avoided
    - 📱 +2h battery life
    """)


if __name__ == '__main__':
    main()