import torch
from torch.utils.data import Sampler


class DomainBatchSampler(Sampler):
    """
    Batch sampler that builds domain-aware minibatches: each minibatch is anchored on one
    batch/domain, with `p_intra_domain` of its cells drawn from that domain and the rest
    from all other domains. Every domain gets `len(domain) // batch_size` minibatches (at
    least one), and all minibatches are shuffled together. A new set of minibatches is
    drawn every epoch.
    """

    def __init__(self, domain_ids, batch_size, p_intra_domain=0.7, min_batch_size=4, device=None):
        """
        Args:
            domain_ids (torch.Tensor): Integer domain id per cell, shape (n_cells,).
            batch_size (int): Cells per minibatch.
            p_intra_domain (float, optional): Fraction of each minibatch drawn from the
                anchor domain.
            min_batch_size (int, optional): Minibatches with fewer distinct cells than this
                are dropped.
            device (torch.device, optional): Device for the index tensors. Defaults to GPU
                if available.
        """
        self.domain_ids = domain_ids
        self.batch_size = batch_size
        self.p_intra_domain = p_intra_domain
        self.min_batch_size = min_batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.unique_domains, self.domain_counts = torch.unique(self.domain_ids, return_counts=True)
        self.valid_batches = None

    @staticmethod
    def _draw(indices, num_batches, count):
        """
        Draws a (num_batches, count) matrix of cells from `indices`: without replacement
        when there are enough cells, otherwise with replacement.
        """
        if len(indices) >= num_batches * count:
            return indices[torch.randperm(len(indices))[:num_batches * count]].view(num_batches, count)
        return indices[torch.randint(len(indices), (num_batches, count))]

    def _generate_batches(self):
        """Draws one epoch of minibatches (lists of cell indices), in random order."""
        in_domain_count = int(self.p_intra_domain * self.batch_size)
        out_domain_count = self.batch_size - in_domain_count

        all_batches = []
        for domain, count in zip(self.unique_domains, self.domain_counts):
            num_batches = max(1, (count // self.batch_size).item())

            domain_indices = torch.where(self.domain_ids == domain)[0]
            out_domain_indices = torch.where(self.domain_ids != domain)[0]

            batch_in_domain = self._draw(domain_indices, num_batches, in_domain_count)
            if len(out_domain_indices) > 0:
                batch_out_domain = self._draw(out_domain_indices, num_batches, out_domain_count)
            else:
                batch_out_domain = torch.empty(num_batches, 0, dtype=torch.long, device=self.device)

            for batch in torch.cat((batch_in_domain, batch_out_domain), dim=1):
                # Drawing with replacement can repeat cells; keep each cell once.
                valid_batch = batch.unique()
                if len(valid_batch) >= self.min_batch_size:
                    all_batches.append(valid_batch)

        # Shuffle all batches to ensure random order of domains
        order = torch.randperm(len(all_batches)).tolist()
        return [all_batches[i] for i in order]

    def __iter__(self):
        self.valid_batches = self._generate_batches()
        for batch in self.valid_batches:
            yield batch.tolist()

    def __len__(self):
        if self.valid_batches is None:
            self.valid_batches = self._generate_batches()
        return len(self.valid_batches)
