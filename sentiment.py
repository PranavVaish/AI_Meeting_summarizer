import nltk
import ssl
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from collections import Counter

def initialize_nltk():
    """
    Initialize NLTK by downloading required resources safely.
    """
    try:
        # Handle SSL certificate issues (e.g., on macOS)
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download 'punkt' if not already available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        # Download VADER lexicon if not already available
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)

    except Exception as e:
        print(f"Warning during NLTK initialization: {str(e)}")

def analyze_sentiment(transcript):
    """
    Analyze sentiment of text using NLTK's VADER sentiment analyzer.

    Args:
        transcript (str): The transcript text to analyze.

    Returns:
        dict: A dictionary containing sentiment analysis results.
    """
    initialize_nltk()

    if not isinstance(transcript, str):
        return {
            "Overall Sentiment": "Invalid input",
            "Error": "Input must be a string"
        }

    transcript = transcript.strip()
    if not transcript:
        return {
            "Overall Sentiment": "Neutral",
            "Positive": "0.0%",
            "Negative": "0.0%",
            "Neutral": "100.0%",
            "Compound Score": 0.0,
            "Note": "Empty input provided"
        }

    try:
        sia = SentimentIntensityAnalyzer()

        try:
            sentences = sent_tokenize(transcript)
        except Exception:
            # Fallback if tokenization fails
            sentences = [s.strip() for s in transcript.split('.') if s.strip()]

        if not sentences:
            return {
                "Overall Sentiment": "Neutral",
                "Positive": "0.0%",
                "Negative": "0.0%",
                "Neutral": "100.0%",
                "Compound Score": 0.0,
                "Note": "No sentences detected"
            }

        scores = []
        sentiments = []

        for sentence in sentences:
            try:
                sentiment_score = sia.polarity_scores(sentence)
                scores.append(sentiment_score)
                compound = sentiment_score['compound']
                if compound >= 0.05:
                    sentiments.append('POSITIVE')
                elif compound <= -0.05:
                    sentiments.append('NEGATIVE')
                else:
                    sentiments.append('NEUTRAL')
            except Exception as e:
                print(f"Sentiment analysis failed on a sentence: {e}")

        if not scores:
            return {
                "Overall Sentiment": "Analysis failed",
                "Note": "No sentiment scores could be calculated"
            }

        # Aggregate results
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        positive = sentiment_counts.get('POSITIVE', 0)
        negative = sentiment_counts.get('NEGATIVE', 0)
        neutral = sentiment_counts.get('NEUTRAL', 0)

        avg_compound = sum(score['compound'] for score in scores) / total
        avg_pos = sum(score['pos'] for score in scores) / total
        avg_neg = sum(score['neg'] for score in scores) / total
        avg_neu = sum(score['neu'] for score in scores) / total

        if avg_compound >= 0.05:
            overall = "Positive"
        elif avg_compound <= -0.05:
            overall = "Negative"
        else:
            overall = "Neutral"

        result = {
            "Overall Sentiment": overall,
            "Positive": f"{positive/total:.1%}",
            "Negative": f"{negative/total:.1%}",
            "Neutral": f"{neutral/total:.1%}",
            "Compound Score": round(avg_compound, 2),
            "Stats": {
                "Total Sentences": total,
                "Positive Sentences": positive,
                "Negative Sentences": negative,
                "Neutral Sentences": neutral,
                "Average Scores": {
                    "Positive": round(avg_pos, 3),
                    "Negative": round(avg_neg, 3),
                    "Neutral": round(avg_neu, 3)
                }
            }
        }

        return result

    except Exception as e:
        return {
            "Overall Sentiment": "Analysis failed",
            "Error": str(e),
            "Note": "Exception raised during sentiment analysis"
        }

# Test block (optional)
if __name__ == "__main__":
    transcript = """
    The team was very enthusiastic about the new project. Everyone contributed great ideas,
    although there were some concerns about the timeline being too tight. Despite the challenges,
    the team agreed to move forward with the plan. The client's requirements were clear and reasonable.
    """
    from pprint import pprint
    pprint(analyze_sentiment(transcript))