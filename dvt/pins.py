"""The revision of each model the annotators load from the Hugging Face hub.

A loader names its revision and the fetch downloads the same one, so a table
made next year comes from the weights that made it this year.
"""

HUB = {
    "facebook/dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    "google/siglip2-so400m-patch16-naflex": "cc24074f717b612951c2dead130904ab9b65a81e",
    "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
}
