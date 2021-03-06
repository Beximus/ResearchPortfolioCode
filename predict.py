# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import cog
# import torch

class Predictor(cog.Predictor):
    def setup(self):
      """Load the model into memory to make running multiple predictions efficient"""
      # self.model = torch.load("./weights.pth")

    @cog.input("query", type=str, help="text to input")
    def predict(self, image):
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_input)
        # return post_processing(output)
