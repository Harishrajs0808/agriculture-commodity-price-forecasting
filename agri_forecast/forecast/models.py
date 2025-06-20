from django.db import models

class CommodityPrice(models.Model):
    commodity_name = models.CharField(max_length=100)
    date = models.DateField()
    price = models.FloatField()

    def __str__(self):
        return f"{self.commodity_name} - {self.date}: ₹{self.price}"
