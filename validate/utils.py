from termcolor import colored
def rprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'white', 'on_red', attrs=["bold"]))

def yprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'yellow',attrs=["bold"]))

def gprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'green',attrs=["bold"]))

def draw_histogram(predictions, targets, save_path):
    import matplotlib
    import matplotlib.pyplot as plt
    from pathlib import Path
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    matplotlib.use('Agg') 
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(predictions, bins=20, alpha=0.5, color='blue', label='Predictions')
    plt.title('Predictions')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(targets, bins=20, alpha=0.5, color='red', label='Targets')
    plt.title('Targets')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)