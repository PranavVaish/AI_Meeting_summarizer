import nltk
# To upgrade pip, run the following command in your terminal or command prompt:
# python -m pip install --upgrade pip
import ssl
import heapq
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def initialize_nltk():
    """
    Initialize NLTK by downloading required resources safely
    """
    try:
        # Handle SSL certificate verification issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
            
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Warning: NLTK initialization issue: {str(e)}")

def summarize_text(transcript, num_sentences=5):
    """
    Summarize text using extractive summarization with NLTK
    
    Args:
        transcript (str): The transcript text to summarize
        num_sentences (int): Number of sentences to include in summary
        
    Returns:
        str: A bullet-point summary of the transcript
    """
    # Initialize NLTK resources
    initialize_nltk()
    
    # Ensure transcript is a string
    if not isinstance(transcript, str):
        return "• Invalid transcript format"
    
    # Clean the transcript
    transcript = transcript.strip()
    if not transcript:
        return "• No transcript content to summarize"
    
    try:
        # Tokenize the text into sentences
        sentences = sent_tokenize(transcript)
        
        # If text is too short, return it as is
        if len(sentences) <= num_sentences:
            return "• " + "\n• ".join(sentences)
        
        # Get English stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except Exception:
            # Fallback if stopwords aren't available
            stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
                             "you", "your", "yours", "yourself", "yourselves", "he", "him", 
                             "his", "himself", "she", "her", "hers", "herself", "it", "its", 
                             "itself", "they", "them", "their", "theirs", "themselves", "what", 
                             "which", "who", "whom", "this", "that", "these", "those", "am", 
                             "is", "are", "was", "were", "be", "been", "being", "have", "has", 
                             "had", "having", "do", "does", "did", "doing", "a", "an", "the", 
                             "and", "but", "if", "or", "because", "as", "until", "while", "of", 
                             "at", "by", "for", "with", "about", "against", "between", "into", 
                             "through", "during", "before", "after", "above", "below", "to", 
                             "from", "up", "down", "in", "out", "on", "off", "over", "under", 
                             "again", "further", "then", "once", "here", "there", "when", 
                             "where", "why", "how", "all", "any", "both", "each", "few", 
                             "more", "most", "other", "some", "such", "no", "nor", "not", 
                             "only", "own", "same", "so", "than", "too", "very", "s", "t", 
                             "can", "will", "just", "don", "should", "now"])
        
        # Tokenize words and create frequency table
        word_frequencies = {}
        
        for sentence in sentences:
            # Handle potential tokenization errors
            try:
                words = word_tokenize(sentence.lower())
            except Exception:
                words = sentence.lower().split()
                
            for word in words:
                if word not in stop_words and word.isalnum():
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1
        
        # Check if we have meaningful word frequencies
        if not word_frequencies:
            # Simple fallback approach
            step = max(1, len(sentences) // num_sentences)
            selected_indices = list(range(0, len(sentences), step))[:num_sentences]
            summary_sentences = [sentences[i] for i in selected_indices]
            return "• " + "\n• ".join(summary_sentences)
        
        # Normalize word frequencies
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency
            
        # Calculate sentence scores
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            # Handle potential tokenization errors
            try:
                words = word_tokenize(sentence.lower())
            except Exception:
                words = sentence.lower().split()
                
            for word in words:
                if word in word_frequencies:
                    sentence_scores[i] = sentence_scores.get(i, 0) + word_frequencies[word]
        
        # Check if we have sentence scores
        if not sentence_scores:
            # Simple fallback approach
            selected_indices = [0]  # Always include first sentence
            if len(sentences) > 1:
                selected_indices.extend([i for i in range(1, len(sentences)) 
                                        if i % (len(sentences) // (num_sentences-1) + 1) == 0][:num_sentences-1])
            summary_sentences = [sentences[i] for i in selected_indices]
            return "• " + "\n• ".join(summary_sentences)
        
        # Get top sentences
        top_sentence_indices = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        top_sentence_indices.sort()  # Sort to maintain original order
        
        # Create bullet point summary
        summary_sentences = [sentences[i] for i in top_sentence_indices]
        summary = "• " + "\n• ".join(summary_sentences)
        
        return summary
            
    except Exception as e:
        # Fallback to simple summarization if processing fails
        try:
            # Try to break text into sentences
            sentences = transcript.split('. ')
            if len(sentences) <= 3:
                return "• " + "\n• ".join(sentences)
            
            # Simple fallback - take first sentence and a few spread throughout
            selected = [sentences[0]]  # Always include first sentence
            step = max(1, len(sentences) // (num_sentences - 1))
            for i in range(1, num_sentences - 1):
                idx = i * step
                if idx < len(sentences):
                    selected.append(sentences[idx])
            if len(sentences) > 1:
                selected.append(sentences[-1])  # Include last sentence
                
            summary = "• " + "\n• ".join(selected)
            return summary
        except:
            # Ultra fallback mode - just return the first part of the text
            if len(transcript) > 500:
                return f"• {transcript[:500]}..."
            else:
                return f"• {transcript}"

if __name__ == "__main__":
                    # Example usage of the summarization function
                    def get_transcript_from_file(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                return file.read()
                        except Exception as e:
                            print(f"Error reading file: {e}")
                            return None

                    # Example 1: Direct text input
                    sample_transcript = """
                    The team discussed project deadlines and assigned tasks. Alice will be handling the front-end development,
                    while Bob will be responsible for the back-end. We agreed that the prototype should be ready by next Friday.
                    Carol raised concerns about the timeline, suggesting we might need additional resources. Dave provided an update
                    on the client's feedback, which was generally positive but included requests for minor interface changes.
                    The meeting concluded with a decision to schedule daily stand-ups to track progress more effectively.
                    """
                    print("Summary of sample text:")
                    print(summarize_text(sample_transcript))
                    print("\n" + "="*50 + "\n")

                    # Example 2: File input
                    file_path = input("Enter path to transcript file (or press Enter to skip): ").strip()
                    if file_path:
                        transcript = get_transcript_from_file(file_path)
                        if transcript:
                            num_sentences = int(input("Enter number of sentences for summary (default=5): ") or 5)
                            print("\nSummary of file content:")
                            print(summarize_text(transcript, num_sentences))