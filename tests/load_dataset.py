from datasets import load_dataset

def load_dataset(name, split='train'):
    """
    Loads the EVE open-ended and hardest-50 Q&A datasets.

    Returns:
        tuple: A tuple containing the train splits of both datasets:
            - qa_eve_open_ended
            - qa_hardest_50_qna
    """
    qa_eve_open_ended = load_dataset(name, split='train')
    return qa_eve_open_ended, qa_hardest_50_qna

# Usage Example
# qa = load_dataset('eve-esa/eve-is-open-ended')
