# feedback_module.py

import dspy

class FeedbackSignature(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.InputField()
    feedback = dspy.InputField()

class FeedbackModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process_feedback = dspy.ChainOfThought(FeedbackSignature)

    def forward(self, question, answer, feedback):
        return self.process_feedback(question=question, answer=answer, feedback=feedback)
