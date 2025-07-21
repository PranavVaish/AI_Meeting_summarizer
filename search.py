import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk
import ssl

def initialize_nltk():
    """Ensure NLTK data is downloaded"""
    try:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
            
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Warning: NLTK initialization issue: {str(e)}")

class MeetingSearch:
    def __init__(self):
        """Initialize the search engine"""
        initialize_nltk()
        self.vectorizer = None  # Initialize when indexing
        self.transcript = None
        self.sentences = []
        self.sentence_vectors = None
    
    def index_transcript(self, transcript):
        """
        Process and index the transcript for searching
        
        Args:
            transcript (str): The meeting transcript text
            
        Returns:
            bool: True if indexing was successful
        """
        # Validate input
        if not isinstance(transcript, str) or not transcript.strip():
            return False
        
        self.transcript = transcript.strip()
        
        # Tokenize the transcript into sentences
        try:
            self.sentences = sent_tokenize(self.transcript)
        except Exception:
            # Fallback tokenization approach
            self.sentences = [s.strip() for s in self.transcript.split('.') if s.strip()]
        
        if not self.sentences:
            return False
        
        # Create TF-IDF vectors for sentences
        try:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.sentence_vectors = self.vectorizer.fit_transform(self.sentences)
            return True
        except Exception as e:
            print(f"Indexing error: {str(e)}")
            self.vectorizer = None
            self.sentence_vectors = None
            return False
    
    def search(self, query, num_results=5):
        """
        Search the transcript for relevant sentences
        
        Args:
            query (str): The search query
            num_results (int): Number of results to return
            
        Returns:
            list: List of tuples (sentence, score)
        """
        # Check if search system is ready
        if self.vectorizer is None or self.sentence_vectors is None or not self.sentences:
            return [("No transcript has been indexed yet.", 0.0)]
        
        # Validate query
        if not isinstance(query, str) or not query.strip():
            return [("Invalid search query.", 0.0)]
            
        query = query.strip()
        
        try:
            # Transform query using the same vectorizer
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarity scores using matrix multiplication (dot product)
            # This fixes the potential error with the @ operator
            similarity = np.dot(self.sentence_vectors.toarray(), query_vector.toarray().T).flatten()
            
            # Get top results with non-zero scores
            top_indices = np.argsort(similarity)[-num_results:][::-1]
            
            # Return results with scores
            results = [(self.sentences[i], float(similarity[i])) for i in top_indices if similarity[i] > 0]
            
            if not results:
                return [("No matching results found.", 0.0)]
                
            return results
        except Exception as e:
            return [(f"Search error: {str(e)}", 0.0)]
    
    def get_context(self, sentence_index, window=1):
        """
        Get surrounding context for a search result
        
        Args:
            sentence_index (int): Index of the sentence
            window (int): Number of sentences before and after
            
        Returns:
            str: Context with the target sentence highlighted
        """
        if not self.sentences:
            return "No transcript available"
        
        # Validate index
        if not isinstance(sentence_index, int) or sentence_index < 0 or sentence_index >= len(self.sentences):
            return "Invalid sentence index"
            
        # Determine context window
        start = max(0, sentence_index - window)
        end = min(len(self.sentences), sentence_index + window + 1)
        
        context = []
        for i in range(start, end):
            if i == sentence_index:
                context.append(f"**{self.sentences[i]}**")  # Highlight the matching sentence
            else:
                context.append(self.sentences[i])
                
        return " ".join(context)
    
    def get_sentence_count(self):
        """
        Get the number of sentences in the transcript
        
        Returns:
            int: Number of sentences
        """
        return len(self.sentences)

# Create a singleton instance
meeting_search = MeetingSearch()

def search_meeting(query, num_results=5):
    """
    Search function to be called from other modules
    
    Args:
        query (str): Search query
        num_results (int): Number of results to return
        
    Returns:
        list: List of tuples (sentence, score)
    """
    return meeting_search.search(query, num_results)

def index_transcript(transcript):
    """
    Index a transcript for searching
    
    Args:
        transcript (str): Meeting transcript
        
    Returns:
        bool: True if indexing was successful
    """
    return meeting_search.index_transcript(transcript)

def get_context(sentence_index, window=1):
    """
    Get context around a sentence
    
    Args:
        sentence_index (int): Index of the sentence
        window (int): Number of sentences before and after
        
    Returns:
        str: Context with the target sentence highlighted
    """
    return meeting_search.get_context(sentence_index, window)

def get_sentence_count():
    """
    Get the number of sentences in the indexed transcript
    
    Returns:
        int: Number of sentences
    """
    return meeting_search.get_sentence_count()

# For testing
if __name__ == "__main__":
    test_transcript = """
    The team discussed project deadlines and assigned tasks. Alice will be handling the front-end development,
    while Bob will be responsible for the back-end. We agreed that the prototype should be ready by next Friday.
    Carol raised concerns about the timeline, suggesting we might need additional resources. Dave provided an update
    on the client's feedback, which was generally positive but included requests for minor interface changes.
    The meeting concluded with a decision to schedule daily stand-ups to track progress more effectively.
    """
    
    index_transcript(test_transcript)
    
    # Test search
    results = search_meeting("project deadline")
    print("Search results for 'project deadline':")
    for sentence, score in results:
        print(f"- Score: {score:.3f}, Result: {sentence}")