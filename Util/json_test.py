import json


def main():
    with open("test.json", "r") as fd:
        file = fd.read()

    print(file)
    print(json.loads(file))

if __name__ == '__main__':
    main()