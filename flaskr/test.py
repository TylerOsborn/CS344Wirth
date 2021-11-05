import json
import unittest
import requests
import recommender as recommender


class Tests(unittest.TestCase):

    """ Tests that a valid search query successfully returns a result."""
    def test_valid_search(self):
        query = "pet"
        data = recommender.search_products(query)
        self.assertFalse(data.empty)

    """ Tests that an invalid search query successfully returns a null result."""
    def test_invalid_search(self):
        query = ""
        data = recommender.search_products(query)
        self.assertTrue(data.empty)

    """ Tests that get item given a valid product ID returns a result."""
    def test_valid_get_item(self):
        query = "0-14477"
        data = recommender.get_item(query)
        self.assertFalse(data.empty)

    """ Tests that get item given an invalid product ID returns a null result."""
    def test_invalid_get_item(self):
        query = "-1"
        result = recommender.get_item(query)
        self.assertIsNone(result)

    """ Tests that the recommendation function returns between 1 and 6 recommendations."""
    def test_recommendations(self):
        data = recommender.get_recommendations("0-14477")

        if 1 <= len(data) <= 6:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    """ Tests that get_cluster returns a number within the range of 0 to k."""
    def test_get_cluster(self):
        k = 30
        cluster = recommender.get_cluster("0-14477")

        if 0 <= cluster <= k:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    """ Tests that cluster api returns the correct cluster data when using a get request."""
    def test_cluster_api(self):
        # Get API data
        api_url = "https://wirth344recommender.herokuapp.com/visual_data"
        api_data = requests.get(url=api_url)

        # Get Cluster data
        f = open('cluster_data.json')
        data = json.load(f)
        f.close()

        self.assertEqual(data, api_data.json())

    """ Tests that the silhouette score is a value in the range -1 to 1."""
    def test_silhouette_score(self):
        silhouette = recommender.get_silhouette()

        if -1 <= silhouette <= 1:
            self.assertTrue(True)
        else:
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
