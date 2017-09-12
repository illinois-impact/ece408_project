all: build push

train_labels.data: ../scripts/train_labels.data
test_labels.data: ../scripts/test_labels.data
label_strings.data: ../scripts/label_strings.data
train_images.data: ../scripts/train_images.data
test_images.data: ../scripts/test_images.data


%.data:
	cp -v $< $@

build: train_labels.data train_images.data test_labels.data test_images.data label_strings.data
	quilt build pearson/ece408 build.yml

push: build
	quilt push pearson/ece408
	rm -rf quilt_packages
	rm -f *.data
