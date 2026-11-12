"""
This module defines the database models for the application using
 SQLAlchemy's ORM.

It includes the `Base` class for declarative mapping and the `RentApartment`
class representing the rent apartments table in the database.
The module uses the SQLAlchemy ORM to map Python classes to database tables.
The structure and fields of the RentApartment class correspond to the columns
in the rent apartments table.
"""

from sqlalchemy import INTEGER, REAL, VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import db_settings


class Base(DeclarativeBase):
    """Base class for declarative mapping of database models."""


class RentApartment(Base):
    """
    SQLAlchemy model class representing the rent apartments table in
    the database.

    Attributes:
        __tablename__ (str): The name of the database table.
        address (Mapped[str]): The address of the apartment (primary key).
        area (Mapped[float]): The area of the apartment in square meters.
        constraction_year (Mapped[int]): The year the apartment was
        constructed.
        rooms (Mapped[int]): The number of rooms in the apartment.
        bedrooms (Mapped[int]): The number of bedrooms in the apartment.
        bathrooms (Mapped[int]): The number of bathrooms in the apartment.
        balcony (Mapped[str]): Indicates if the apartment has a balcony.
        storage (Mapped[str]): Indicates if the apartment has storage space.
        parking (Mapped[str]): Indicates if the apartment has parking space.
        furnished (Mapped[str]): Indicates if the apartment is furnished.
        garage (Mapped[str]): Indicates if the apartment has a garage.
        garden (Mapped[str]): Indicates if the apartment has a garden.
        energy (Mapped[str]): Energy efficiency rating of the apartment.
        facilities (Mapped[str]): Additional facilities available in the
        apartment.
        zip (Mapped[str]): ZIP code of the apartment's location.
        neighborhood (Mapped[str]): Neighborhood where the apartment is
        located.
        rent (Mapped[int]): Monthly rent for the apartment.
    """

    __tablename__ = db_settings.rent_apartments_table_name

    address: Mapped[str] = mapped_column(VARCHAR(), primary_key=True)
    area: Mapped[float] = mapped_column(REAL())
    constraction_year: Mapped[int] = mapped_column(INTEGER())
    rooms: Mapped[int] = mapped_column(INTEGER())
    bedrooms: Mapped[int] = mapped_column(INTEGER())
    bathrooms: Mapped[int] = mapped_column(INTEGER())
    balcony: Mapped[str] = mapped_column(VARCHAR())
    storage: Mapped[str] = mapped_column(VARCHAR())
    parking: Mapped[str] = mapped_column(VARCHAR())
    furnished: Mapped[str] = mapped_column(VARCHAR())
    garage: Mapped[str] = mapped_column(VARCHAR())
    garden: Mapped[str] = mapped_column(VARCHAR())
    energy: Mapped[str] = mapped_column(VARCHAR())
    facilities: Mapped[str] = mapped_column(VARCHAR())
    zip: Mapped[str] = mapped_column(VARCHAR())
    neighborhood: Mapped[str] = mapped_column(VARCHAR())
    rent: Mapped[int] = mapped_column(INTEGER())
