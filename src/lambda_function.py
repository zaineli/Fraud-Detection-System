# src/lambda_function.py

from inference import predict_fraud

def lambda_handler(event, context):
    try:
        input_dict = event.get("body") if "body" in event else event
        if isinstance(input_dict, str):
            import json
            input_dict = json.loads(input_dict)

        result = predict_fraud(input_dict)
        return {
            "statusCode": 200,
            "body": result
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }
