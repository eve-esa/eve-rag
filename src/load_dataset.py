from datasets import load_dataset

def load_dataset(name, split='train'):
    """
    Loads the EVE open-ended and hardest-50 Q&A datasets.

    Returns:
        List: A list containing the disctionaries of q/a:

    """
    qa = load_dataset(name, split=split)
    return qa

# Usage Example
# qa = load_dataset('eve-esa/eve-is-open-ended')
