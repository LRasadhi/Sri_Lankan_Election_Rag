from utils.translation import translate_text
import argparse

def test_direct_translation():
    """Test translation function directly"""
    print("\nTesting direct translation:")
    test_text = "Hello! This is a test message. How are you today?"
    print(f"\nOriginal text:\n{test_text}")
    
    translated = translate_text(test_text)
    print(f"\nTranslated to Sinhala:\n{translated}")

def test_app_translation():
    """Test translation through main app"""
    from app import answer_question, initialize_rag_system
    
    print("\nTesting RAG system with translation:")
    _, _, _, rag_chain = initialize_rag_system()
    
    test_question = "What is proportional representation?"
    result = answer_question(test_question, rag_chain, translate_to_sinhala=True)
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nEnglish Answer:\n{result['english_answer']}")
    print(f"\nSinhala Translation:\n{result['sinhala_answer']}")
    print("\nSources:")
    for src in result['sources']:
        print(f"- {src}")

def main():
    parser = argparse.ArgumentParser(description='Test translation functionality')
    parser.add_argument('--mode', choices=['direct', 'rag', 'both'], 
                      default='both', help='Test mode to run')
    
    args = parser.parse_args()
    
    if args.mode in ['direct', 'both']:
        test_direct_translation()
    
    if args.mode in ['rag', 'both']:
        test_app_translation()

if __name__ == "__main__":
    main()