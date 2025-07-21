import uuid

def get_session_id():
    """
    Generate a unique session ID for chat history tracking.
    In a real application, this might come from a user session or other identifier.
    """
    return str(uuid.uuid4())