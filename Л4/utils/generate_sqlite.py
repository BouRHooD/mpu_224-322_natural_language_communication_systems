import sqlite3

class SqliteInteraction():
    def __init__(self, base):
        self.connection = sqlite3.connect(base)
        self.cursor = self.connection.cursor()

    def select(self, table):
        return self.cursor.execute(f"SELECT * FROM {table}")

    def update(self, table, params, whereCol, whereVal):
        for i in params:
            print(f"UPDATE {table} SET {i} = {params[i]} WHERE {whereCol} = {whereVal}")
            self.cursor.execute(f"UPDATE {table} SET {i} = {params[i]} WHERE {whereCol} = ?", (whereVal,))
            self.connection.commit()

    def selectWhere(self, table, params):
        res = []
        for i in params:
            res.append(self.cursor.execute(f"SELECT * FROM {table} where {i} = ?", (params[i],)).fetchall())
        return res
