# ECR Scoring Rubric (portable, unparsed)

You are estimating the Engagement Continuation Rate (ECR) of a short video: the probability that a viewer keeps watching past 5 seconds. Follow the rubric below. Score each dimension in [0, 1], then apply the scoring formula, then output ECR in [0, 100].

## Raw rubric text

{
  "dimensions": [
    {
      "name": "Opening-second hook",
      "what_to_look_for": "Engaging opening shot or frame that draws the viewer in",
      "direction": "positive",
      "weight": 0.3,
      "detection_heuristic": "Aesthetic quality, motion, and interest"
    },
    {
      "name": "Visual novelty",
      "what_tolook_for": "New and interesting visual elements that keep the viewer's attention",
      "direction": "positive",
      "weight": 0.2,
      "detectionheuristic": "Change of scene, color, texture, object"
    },
    {
      "name": "Motion/pace",
      "what_to_look_for": "Sensory dynamism and movement that conveys a sense of time and place",
      "direction": " positive",
      " weight": 0.2,
      " detectionheuristic": " Change of motion, sound, and rhythm"
    },
    {
      " name": "Subject clarity",
      " what to look for": "Clear and unambiguous subject matter that is easy to understand",
      " direction": " positive",
      " weight": 0.4,
      " detection heuristic": " Text overlay, object count, and object distinctness"
    },
    {
      " name": "Cognitive payoff/curiosity gap",
      " what to look for": "An intriguing aspect of the content that invites further exploration",
      " direction": " positive",
      " weight": 0.1,
      " detection heuristic": " Unfamiliarity, open-endedness, and ambiguity"
    },
    {
      " name": "Aesthetic quality",
      " what to look for": "Aesthetically pleasing elements that evoke positive emotions",
      " direction": " positive",
      " weight": 0.1,
      " detection heuristic": " Color palette, texture, and form"
    },
    {
      " name": "Text overlay strategy",
      " what to look for": "The use of text to complement the visual content",
      " direction": " positive",
      " weight": 0.1,
      " detection heuristic": " Text placement, font, and legibility"
    },
    {
      " name": "Title-content congruence",
      " what to look for": "The title accurately reflects the content of the video",
      " direction": " positive",
      " weight": 0.1,
      " detection heuristic": " Title length, word count, and use of capital letters"
    },
    {
      " name": "Emotional valence/arousal",
      " what to look for": " The emotional tone of the video, conveyed through visuals, sounds, and text",
      " direction": " positive",
      " weight": 0.1,
      " detection heuristic": " Color, object count, and object distinctness"
    },
    {
      " name": "Familiar vs. novel framing",
      " what to look for": " The balance between familiar and novel elements in the video",
      " direction": " positive",
      " weight": 0.1,
      " detection heuristic": " Familiarity of objects, setting, and theme"
    },
    {
      " name": "Platform/platform fit",
      " what to look for": " The video's suitability for the platform it is being viewed on",
      " direction": " positive",
      " weight": 0.1,
      " detection heuristic": " Screen size, aspect ratio, and video orientation"
    }
  ],
  "scoring formula": "The ECR score is the sum of the weighted scores for each dimension, with the exception of the text overlay strategy, which is only considered if the video has text overlay. The final score is then normalized to [0,100].",
  "caveats": [
    "The text overlay strategy is only relevant for videos with text overlay.",
    "The emotional valence/arousal is only relevant for videos with a clear emotional tone."
  ]
}