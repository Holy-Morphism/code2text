import streamlit as st
from main import text
def main():
    st.title("Code to Text Generator")
    
    sentence = st.text_area("Enter the code snippet:")
    
    if st.button("Generate Text"):
        if sentence:
            with st.spinner("Generating text..."):
                generated_text = text(sentence, "./latest_checkpoint.pt", "Salesforce/codet5-small", "bert-base-uncased", 300)
                st.success("Text generated successfully!")
                st.text_area("Generated Text:", generated_text, height=200)
        else:
            st.error("Please fill in the field.")

if __name__ == "__main__":
    main()
