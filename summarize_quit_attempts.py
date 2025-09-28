import pandas as pd
from scipy.stats import ttest_ind

def summarize_quit_attempts(file_path):
    """
    Generates a summary of participants’ quit attempt behaviour before and after the MIBot conversation.

    Args:
        file_path (str): The path to the CSV file containing the data.
    """
    try:
        df = pd.read_csv(file_path)

        # Before conversation
        pre_convo_quit_attempts = df['PreConvoQuitAttempt'].sum()
        pre_convo_total_attempts = df['PreConvoNumQuitAttempts'].sum()
        
        # After conversation
        post_convo_quit_attempts = df['WeekLaterQuitAttempt'].sum()
        post_convo_total_attempts = df['WeekLaterNumQuitAttempts'].sum()

        num_participants = len(df)

        print("Summary of Quit Attempt Behaviour")
        print("=" * 35)
        print(f"Total number of participants: {num_participants}")
        print("\n--- Before MIBot Conversation ---")
        print(f"Participants who attempted to quit: {pre_convo_quit_attempts} ({pre_convo_quit_attempts/num_participants:.2%})")
        print(f"Total number of quit attempts: {pre_convo_total_attempts}")
        if pre_convo_quit_attempts > 0:
            print(f"Average quit attempts per participant who tried: {pre_convo_total_attempts/pre_convo_quit_attempts:.2f}")

        print("\n--- After MIBot Conversation (1 Week Later) ---")
        print(f"Participants who attempted to quit: {post_convo_quit_attempts} ({post_convo_quit_attempts/num_participants:.2%})")
        print(f"Total number of quit attempts: {post_convo_total_attempts}")
        if post_convo_quit_attempts > 0:
            print(f"Average quit attempts per participant who tried: {post_convo_total_attempts/post_convo_quit_attempts:.2f}")
        
        print("\n--- Change in Quit Attempts ---")
        change_in_attempts = post_convo_quit_attempts - pre_convo_quit_attempts
        print(f"Change in number of participants attempting to quit: {change_in_attempts}")

        # Confidence gain analysis
        df['ConfidenceGain'] = df['PostRulerConfidence'] - df['PreRulerConfidence']
        
        quit_attempters = df[df['WeekLaterQuitAttempt'] == True]
        non_attempters = df[df['WeekLaterQuitAttempt'] == False]

        mean_confidence_gain_attempters = quit_attempters['ConfidenceGain'].mean()
        mean_confidence_gain_non_attempters = non_attempters['ConfidenceGain'].mean()

        ttest_result = ttest_ind(quit_attempters['ConfidenceGain'], non_attempters['ConfidenceGain'])

        print("\n--- Confidence Gain Analysis ---")
        print(f"Mean confidence gain for quit attempters: {mean_confidence_gain_attempters:.2f}")
        print(f"Mean confidence gain for non-attempters: {mean_confidence_gain_non_attempters:.2f}")
        print(f"p-value from t-test: {ttest_result.pvalue:.4f}")

        if ttest_result.pvalue < 0.05:
            print("The difference in confidence gain between quit attempters and non-attempters is statistically significant.")
        else:
            print("The difference in confidence gain between quit attempters and non-attempters is not statistically significant.")


    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    summarize_quit_attempts('/Users/zafarmahmood/Desktop/src/MIBOT_ACL2025/data.csv')
