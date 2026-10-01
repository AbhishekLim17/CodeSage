"""User management service."""

import re
from dataclasses import dataclass

MAX_NAME_LENGTH = 50
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+$")


@dataclass
class User:
    """A registered user."""

    name: str
    email: str


# Raised when input validation fails.
class ValidationError(Exception):
    pass


class UserService:
    """Creates and looks up users."""

    def __init__(self, repository):
        self.repository = repository

    # Validate then persist a new user.
    def create_user(self, name, email):
        if len(name) > MAX_NAME_LENGTH:
            raise ValidationError("name too long")
        if not EMAIL_RE.match(email):
            raise ValidationError("invalid email")
        user = User(name=name, email=email)
        self.repository.save(user)
        return user

    @staticmethod
    def normalize(email):
        return email.strip().lower()

    async def find_by_email(self, email):
        return self.repository.get(self.normalize(email))


def make_service(repository):
    """Factory used by the app."""
    return UserService(repository)


if __name__ == "__main__":
    print(make_service(None))
