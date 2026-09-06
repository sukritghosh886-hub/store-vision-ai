from models.retail_assistant import (
    RetailAssistant
)


assistant = RetailAssistant()


assistant.register_visitor()


for _ in range(20):

    assistant.observe_product(
        "Mathematics Book"
    )


for _ in range(8):

    assistant.register_interaction(
        "Mathematics Book"
    )


assistant.register_purchase(
    "Mathematics Book"
)


for _ in range(15):

    assistant.observe_product(
        "Water Bottle"
    )


for _ in range(6):

    assistant.register_interaction(
        "Water Bottle"
    )


print(
    assistant.summary()
)