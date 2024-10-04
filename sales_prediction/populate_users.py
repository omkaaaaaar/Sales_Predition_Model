from .models import Forgot_User  # Replace 'your_app' with your actual app name

def populate_forgot_users():
    user_data = [
        {"username": "samarth", "fullname": "Samarth Jadhav", "email": "samarth@example.com", "phone": "1234567895", "role": "Employee", "is_staff": True, "password": "pbkdf2_sha256$720000$8OmjTLXeAXzvjDwoL4YIiM$C3TmFbUfT3k49C+UZfSLkKPvFd9RNp2a38QyYmhOo14="},
        {"username": "tushar", "fullname": "Tushar Deshmukh", "email": "tushar@example.com", "phone": "1234567894", "role": "Employee", "is_staff": True, "password": "pbkdf2_sha256$720000$XKEitoUsX9mw2VL2RCx31R$8MfVYcpbFh28/mWzb2OCv1x+/reBrMvTuSniHvBUwAk="},
        {"username": "soham", "fullname": "Soham Mhatre", "email": "soham@example.com", "phone": "1234567893", "role": "Employee", "is_staff": True, "password": "pbkdf2_sha256$720000$PSRWED1qzAfpmAmXiNXgsN$8KUe3Xsd1URjEf5iJXAidwDHVfUCrLZ/oLo5mwlw//o="},
        {"username": "omkar", "fullname": "Omkar Patker", "email": "omkar@example.com", "phone": "1234567892", "role": "Developer", "is_staff": True, "password": "pbkdf2_sha256$720000$eDCb6bPpLdNmreJjRSXjMY$I0svPs0Du23B5/vBSQdqxV5l8uy+8meRJBbu+5oss84="},
        {"username": "sahil", "fullname": "Sahil Pagar", "email": "sahil@example.com", "phone": "1234567891", "role": "Developer", "is_staff": True, "password": "pbkdf2_sha256$720000$6nhO8bMOQbT0wopScGlHJq$9xoVD2sqpiHkwrxQe0rG+eDYxqWy6L+Wsyl6U0boy2c="},
        {"username": "krunal", "fullname": "Krunal Gurule", "email": "krunal@example.com", "phone": "1234567890", "role": "Developer", "is_staff": True, "password": "pbkdf2_sha256$720000$d6D7yFyi5kXa0JetlSOSQH$Wr/YVmh0e7Bx3Ct1i2+O716KtBJxVADdxMVOmKLUK8w="},
        {"username": "nikhil1", "fullname": "Nikhil Sutar", "email": "mhtcet98@gmail.com", "phone": "7738544966", "role": "Admin", "is_staff": True, "password": "pbkdf2_sha256$720000$H6SSxyLiTmFlMsqfSSrv7t$izGos/mxHzQDYthL5EtZv6cChtMHcD09bpIFKDXppn0="}
    ]

    for user_info in user_data:
        user = Forgot_User.objects.create(
            username=user_info['username'],
            fullname=user_info['fullname'],
            email=user_info['email'],
            phone=user_info['phone'],
            role=user_info['role'],
            is_staff=user_info['is_staff'],
        )
        user.password = user_info['password']  # Directly assigning hashed password
        user.save()

    print("Users have been populated successfully.")

if __name__ == "__main__":
    populate_forgot_users()
