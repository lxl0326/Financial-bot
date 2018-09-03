# python3.6

from random import choice
from datetime import datetime
import matplotlib.pyplot as plt

from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config

from iexfinance import Stock
from iexfinance import get_historical_data


# define some synonyms to make it more robust
# return values which can be directly used in iexfinance structure
# I only found the symbol of baidu and tesla
def synonyms(word):
    if word.lower() in ["baidu", "beidu", "bidu"]:
        return "BIDU"

    if word.lower() in ["telsa", "tesla", "tsla", "tlsa"]:
        return "TSLA"

    return word


def create_entities():
    entities = {}

    for dict in interpretation["entities"]:
        entities[dict["entity"]] = dict["value"]

    return entities


def create_intent():
    pre_intent = "default"

    if interpretation["intent"]["name"] in template_responds.keys() or "ask":
        pre_intent = interpretation["intent"]["name"]

    if "start_time" in entities.keys():
        intent = "history"

    elif "time_pointer" in entities.keys():
        time_pointer = entities["time_pointer"]
        if time_pointer.lower() in \
                ["now", "current", "present", "at present", "moment", "the moment", "this moment"]:
            intent = "current"
        else:
            intent = "open"

    elif pre_intent == "ask":
        intent = "current"

    else:
        intent = pre_intent

    return intent


def respond(template_responds, intent):
    if intent == "current":
        try:
            price = Stock(synonyms(company)).get_price()
        except IndexError or ValueError:
            responds = choice(template_responds["check"])
        else:
            responds = choice(template_responds[intent]).format(company, price)

    elif intent == "open":
        try:
            price = Stock(synonyms(company)).get_open()
        except IndexError or ValueError:
            responds = choice(template_responds["check"])
        else:
            responds = choice(template_responds[intent]).format(company, price)

    elif intent == "history":
        start_y, start_m, start_d = [int(_) for _ in entities["start_time"].split(".")]
        end_y, end_m, end_d = [int(_) for _ in entities["start_time"].split(".")]

        start = datetime(start_y, start_m, start_d)
        end = datetime(end_y, end_m, end_d)

        df = get_historical_data("AAPL", start=start, end=end, output_format='pandas')
        df.plot()
        plt.show()
        return

    else:
        # print(interpretation)
        responds = choice(template_responds[intent])

    return responds



template_responds = {"greet":["Hi, what can I do for you?",
                     "Hi, sir.",
                     "Hi, I'm ready for you."],

            "goodbye":["See you later.",
                       "Good bye.",
                       "Bye.",
                       "It's very nice of you! Bye.",
                       "It's my pleasure, have a good day!",
                       "Thanks, goodbye"],

            "current":["The current stock price of {} is {}.",      # company, price
                       "The stock price of {} is {} now."],         # company, price

            "open":["The opening price of {} is {}",                # company, price
                    "The stock price of {} is {} this morning."],   # company, price

            "ask_function":["I could help you check stock prices from some famous companies about the current "+
                            "values, opening values and history plot."],

            "default":["Sorry, I don't understand you.",
                       "Sorry, what did you say?"
                       ],

            "check":["Sorry, something goes wrong, please make assure that you inputted the right information.",
                     "Sorry, I can't do that. Please check your information."
                     ]
            }


# Create a trainer that uses this config
trainer = Trainer(config.load("./nlu_config.yml"))

# Load the training data
training_data = load_data('./models/current/nlu/training_data.json')

# Create an interpreter by training the model
interpreter = trainer.train(training_data)


company = ""
while True:
    message = input()
    interpretation = interpreter.parse(message)
    entities = create_entities()
    intent = create_intent()

    if "company" in entities.keys():
        company = entities["company"]

    if intent in ["current", "open", "history"] and company == "":
        responds = respond(template_responds, "check")
    else:
        responds = respond(template_responds, intent)

    # print(interpretation) used for checking code
    print(responds)
    if intent == "goodbye":
        break







