import unittest
from carta_framework.models import TemporalScope
from pydantic import ValidationError

class TestTemporalScope(unittest.TestCase):
    def test_valid_instantiation(self):
        ts = TemporalScope(horizon_timestamp=1000.0)
        self.assertEqual(ts.horizon_timestamp, 1000.0)
        
    def test_invalid_type(self):
        with self.assertRaises(ValidationError):
            TemporalScope(horizon_timestamp="invalid")
